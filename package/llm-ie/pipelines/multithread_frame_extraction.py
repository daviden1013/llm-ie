# This script processes many documents in parallel (multithreading) to extract frames
# Units in each document are concurrently processed using asyncio. Documents are processed in parallel using multiprocessing.
# ** Modify the config and logic as needed for your specific use case. **

import argparse
from easydict import EasyDict
import yaml
import os
from tqdm import tqdm
import logging
from llm_ie import OpenAIInferenceEngine, DirectFrameExtractor, LLMInformationExtractionDocument, WholeDocumentUnitChunker, NoContextChunker
import multiprocessing 


def process_file_worker(filename, text_content, config, system_prompt_str, prompt_template_str):
    """
    Processes a single file: performs extraction and saves the result.
    This function is intended to be run in a separate process.
    """
    try:
        llm = OpenAIInferenceEngine(model="unused", 
                                    api_key="unused", 
                                    base_url=config["base_url"])

        # Initialize extractor components within the worker process
        unit_chunker = WholeDocumentUnitChunker()
        context_chunker = NoContextChunker()
        extractor = DirectFrameExtractor(llm,
                                         unit_chunker=unit_chunker,
                                         context_chunker=context_chunker,
                                         prompt_template=prompt_template_str,
                                         system_prompt=system_prompt_str)

        frames = extractor.extract_frames(text_content=text_content,
                                          concurrent=False)

        llmie = LLMInformationExtractionDocument(doc_id=filename, text=text_content)
        llmie.add_frames(frames, create_id=True)

        output_path = os.path.join(config["out_dir"], config['run_name'], f"{llmie.doc_id}.llmie")
        llmie.save(output_path)

        return filename, "success"
    except Exception as e:
        return filename, f"error: {str(e)}"

def wrap_process_file_worker(args):
    """
    Wrapper function to unpack arguments for the worker function.
    This is necessary for multiprocessing to work correctly with the function.
    """
    return process_file_worker(*args)


def main():
    parser = argparse.ArgumentParser(description="Information extraction pipeline with parallel processing.")
    add_arg = parser.add_argument
    add_arg("-c", "--config", help='Directory to config file', type=str, required=True)
    add_arg("-o", "--overwrite", help="Overwrite existing output files", action="store_true")
    add_arg("--num_workers", help="Number of parallel worker processes. Defaults to CPU count.", type=int, default=os.cpu_count())
    args = parser.parse_known_args()[0]

    """ Load config """
    with open(args.config) as yaml_file:
        config = EasyDict(yaml.safe_load(yaml_file))

    """ Logging """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.info('# Information extraction pipeline starts...')
    logging.info(f"Using {args.num_workers} worker processes.")

    """ Load system prompt """
    logging.info(f"Loading system prompt from {config['system_pormpt_dir']}...")
    with open(config["system_pormpt_dir"], "r") as f:
        system_prompt = f.read()

    """ Load prompt template """
    logging.info(f"Loading prompt template from {config['prompt_temp_dir']}...")
    with open(config['prompt_temp_dir'], 'r') as f:
        prompt_template = f.read()

    """ Ensure output directory exists """
    output_dir = os.path.join(config["out_dir"], config['run_name'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    """ Load data """
    logging.info(f"Loading text from {config['data_dir']}")
    # **
    # Adjust the file loading logic as needed for your data.
    # **
    data = []

    """ Check exist outputs """
    if args.overwrite:
        pass
        # **
        # Adjust the logic as needed for your data.
        #
        # Example:
        #
        # exist = os.listdir(output_dir)
        # logging.info(f"Found {len(exist)} existing outputs.")
        # data = [d for d in data if d['filename'] in exist]
        # logging.info(f"Found {len(data)} files to process after filtering.")
        # **

    logging.info(f"Starting extraction for {len(data)} files...")

    """ Prepare arguments for each worker task """
    tasks_args = [
        (filename, text, config, system_prompt, prompt_template)
        for filename, text in data
    ]

    successful_extractions = 0
    failed_extractions = 0

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(data), desc="Processing files") as pbar:
            for original_filename, status_message in pool.imap_unordered(wrap_process_file_worker, tasks_args):
                if status_message == "success":
                    successful_extractions += 1
                else:
                    failed_extractions += 1
                    logging.warning(f"Processing failed for {original_filename}: {status_message}")
                pbar.update(1)

    logging.info("Extraction pipeline finished.")
    logging.info(f"Successfully processed: {successful_extractions} files.")
    logging.info(f"Failed to process: {failed_extractions} files.")
    logging.info(f"Output saved in: {output_dir}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()