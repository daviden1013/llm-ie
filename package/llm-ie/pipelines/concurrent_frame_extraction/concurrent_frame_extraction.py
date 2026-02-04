import argparse
from easydict import EasyDict
import yaml
import json
import os
from tqdm import tqdm
import time
import logging
from llm_ie import BasicLLMConfig, OpenAIReasoningLLMConfig, VLLMInferenceEngine, SentenceFrameExtractor, LLMInformationExtractionDocument
import asyncio


async def process_file_worker(doc_id, text_content, frame_extractor: SentenceFrameExtractor, config):
    """
    Processes a single file: performs extraction and saves the result.
    This function is intended to be run as an async task.
    """
    try:
        start_time = time.time()
        
        # extract frames
        frames, messages_log = await frame_extractor.extract_frame_async(text_content=text_content, return_messages_log=True)
    
        # Process and save llmie doc
        llmie = LLMInformationExtractionDocument(doc_id=doc_id, text=text_content)
        llmie.add_frames(frames)
        llmie.save(os.path.join(config["out_dir"], config['run_name'], f"{doc_id}.llmie"))
        
        # Save message log
        end_time = time.time() 
        elapsed_time = end_time - start_time 

        # Save message log and processing time
        log_data = { 
            "processing_time_seconds": round(elapsed_time, 2),
            "messages": messages_log
        }
        with open(os.path.join(config["log_dir"], config['run_name'], f"{doc_id}.json"), "w") as f:
            json.dump(log_data, f, indent=4) 

        return doc_id, "success"
    except Exception as e:
        logging.error(f"Error processing {doc_id}: {str(e)}", exc_info=True)
        return doc_id, f"error: {str(e)}"


async def document_worker(worker_id: int,
                         doc_queue: asyncio.Queue,
                         frame_extractor: SentenceFrameExtractor,
                         config,
                         stats: dict,
                         progress_bar: tqdm):
    """
    Worker that continuously processes documents from the queue.
    Each worker handles one document at a time, but multiple workers run concurrently.
    """
    while True:
        try:
            # Get next document (with timeout to detect empty queue)
            doc = await asyncio.wait_for(doc_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Queue is empty, worker can exit
            break
        
        try:
            logging.info(f"[Worker-{worker_id}] Starting document {doc.doc_id}")
            
            result = await process_file_worker(
                str(doc.doc_id),
                doc.text,
                frame_extractor,
                config
            )
            
            doc_id, status = result
            
            if status == "success":
                stats['success'] += 1
                logging.info(f"[Worker-{worker_id}] ✓ Completed {doc_id} ({stats['success'] + stats['failed']}/{stats['total']})")
            else:
                stats['failed'] += 1
                logging.warning(f"[Worker-{worker_id}] ✗ Failed {doc_id}: {status}")
                
        except Exception as e:
            stats['failed'] += 1
            logging.error(f"[Worker-{worker_id}] Exception processing {doc.doc_id}: {e}", exc_info=True)
        finally:
            doc_queue.task_done()
            progress_bar.update(1)


async def main():
    parser = argparse.ArgumentParser(description="Information extraction pipeline with async worker pool processing.")
    add_arg = parser.add_argument
    add_arg("-c", "--config", help='Directory to config file', type=str, required=True)
    add_arg("--overwrite", help="Overwrite existing output files.", action="store_true")
    args = parser.parse_known_args()[0]

    """ Load config """
    with open(args.config) as yaml_file:
        config = EasyDict(yaml.safe_load(yaml_file))

    """ Logging """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.info('# Information extraction pipeline starts...')
    logging.info(f"Using {args.num_workers} async document workers.")

    """ Load system prompt """
    logging.info(f"Loading system prompt from {config['system_prompt_dir']}...")
    with open(config["system_prompt_dir"], "r") as f:
        system_prompt = f.read()

    """ Load prompt template """
    logging.info(f"Loading prompt template from {config['prompt_temp_dir']}...")
    with open(config["prompt_temp_dir"], "r") as f:
        prompt_template = f.read()

    """ Load data """
    logging.info(f"Loading text from {config['data_dir']}")
    filenames = os.listdir(config["data_dir"])
    input_docs = []
    for filename in filenames:
        doc = LLMInformationExtractionDocument(filename=os.path.join(config["data_dir"], filename))
        input_docs.append(doc)

    """ Ensure output and log directories exist """
    output_dir = os.path.join(config["out_dir"], config['run_name'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    log_dir = os.path.join(config["log_dir"], config['run_name'])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        logging.info(f"Created log directory: {log_dir}")

    """ Exclude notes already in output directory """
    if not args.overwrite:
        exist = [f.replace(".llmie", "") for f in os.listdir(output_dir)]
        logging.info(f"Found {len(exist)} files already processed.")
        input_docs = [doc for doc in input_docs if doc.doc_id not in exist]

    if not input_docs:
        logging.info("No documents to process. Exiting.")
        return

    """ Initialize LLM inference engine (shared across all workers) """
    logging.info('Loading inference engine...')
    # OpenAI Azure
    llm = VLLMInferenceEngine(model=config["model"], 
                              config=eval(config["llm_config"]),
                              max_concurrent_requests=config['max_concurrent_requests'])

    """ Initialize extractors (shared across all workers) """
    logging.info('Initializing frame extractors...')
    frame_extractor = SentenceFrameExtractor(llm, prompt_template, system_prompt=system_prompt)

    """ Create document queue and populate it """
    logging.info(f"Creating document queue with {len(input_docs)} documents...")
    doc_queue = asyncio.Queue()
    for doc in input_docs:
        await doc_queue.put(doc)

    """ Statistics tracking """
    stats = {
        'total': len(input_docs),
        'success': 0,
        'failed': 0
    }

    """ Create progress bar """
    progress_bar = tqdm(total=len(input_docs), desc="Processing documents")

    """ Start worker pool """
    logging.info(f"Starting {config['num_workers']} async workers...")
    workers = [
        document_worker(
            worker_id=i,
            doc_queue=doc_queue,
            frame_extractor=frame_extractor,
            config=config,
            stats=stats,
            progress_bar=progress_bar
        )
        for i in range(config['num_workers'])
    ]

    """ Run all workers concurrently """
    await asyncio.gather(*workers)

    """ Cleanup """
    progress_bar.close()

    """ Final statistics """
    logging.info("="*60)
    logging.info("Extraction pipeline finished.")
    logging.info(f"Successfully processed: {stats['success']} files.")
    logging.info(f"Failed to process: {stats['failed']} files.")
    logging.info(f"Output saved in: {output_dir}")
    logging.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())