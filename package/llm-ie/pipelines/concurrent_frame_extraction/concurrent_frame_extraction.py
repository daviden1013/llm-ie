# This script processes many documents in concurrent (asyncio) to extract frames
# Documents and units are concurrently processed using asyncio.
# ** Modify the config and logic as needed for your specific use case. **

import argparse
from easydict import EasyDict
import yaml
import json
import os
from tqdm import tqdm
import logging
import asyncio
from llm_ie import (
    OpenAIInferenceEngine, 
    DirectFrameExtractor, 
    LLMInformationExtractionDocument, 
    WholeDocumentUnitChunker, 
    NoContextChunker
)

async def process_file_worker(filename, text_content, config, frame_extractor):
    """
    Processes a single file: performs structure extraction, applies logic, 
    performs frame extraction, and saves the result.
    """
    try:
        # Extract frames
        frames, messages_log = await frame_extractor.extract_frames_async(
            text_content=text_content,
            concurrent_batch_size=config['concurrent_batch_size'],
            return_messages_log=True
        )

        # Package frames into LLMIE document
        llmie = LLMInformationExtractionDocument(doc_id=filename, text=text_content)
        llmie.add_frames(frames, create_id=True)

        # Save output
        output_path = os.path.join(config["out_dir"], config['run_name'], f"{llmie.doc_id}.llmie")
        llmie.save(output_path)

        # Save messages log if needed
        log_output_path = os.path.join(config["messages_log_dir"], config['run_name'], f"{llmie.doc_id}.json")
        with open(log_output_path, 'w') as log_file:
            json.dump(messages_log, log_file, indent=4)

        return filename, "success"
    except Exception as e:
        return filename, f"error: {str(e)}"


async def main():
    parser = argparse.ArgumentParser(description="Frame extraction pipeline with async processing.")
    add_arg = parser.add_argument
    add_arg("-c", "--config", help='Directory to config file', type=str, required=True)
    add_arg("-o", "--overwrite", help="Overwrite existing output files", action="store_true")
    args = parser.parse_known_args()[0]

    """ Load config """
    with open(args.config) as yaml_file:
        config = EasyDict(yaml.safe_load(yaml_file))

    """ Logging """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.info('# Async Frame extraction pipeline starts...')

    """ Initialize Engine """
    # Using global concurrency control provided by the engine
    logging.info(f"Initializing engine with concurrency limit: {config['max_concurrent_requests']}")
    llm = OpenAIInferenceEngine(
        model="unused", 
        config=config["engine_config"],
        max_concurrent_requests=config["max_concurrent_requests"]
    )

    """ Load prompts """
    logging.info(f"Loading prompts...")
    with open(config["system_prompt_dir"], "r") as f: system_prompt = f.read()
    with open(config["frame_prompt_dir"], "r") as f: frame_prompt = f.read()

    """ Initialize Extractors """
    frame_extractor = DirectFrameExtractor(
        inference_engine=llm,
        unit_chunker=WholeDocumentUnitChunker(),
        context_chunker=NoContextChunker(),
        prompt_template=frame_prompt,
        system_prompt=system_prompt
    )

    """ Ensure output directory exists """
    output_dir = os.path.join(config["out_dir"], config['run_name'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    """ Load data """
    logging.info(f"Loading text from {config['data_dir']}")
    # ** Adjust the file loading logic as needed for your data. **
    data = []
    # Example:
    # for f in os.listdir(config['data_dir']):
    #     with open(os.path.join(config['data_dir'], f), 'r') as file:
    #         data.append((f, file.read()))
    
    # Mock data for demonstration if empty
    if not data:
        logging.warning("No data loaded. Please implement data loading logic.")
    
    """ Check exist outputs """
    if args.overwrite:
        pass
        # ** Adjust logic to filter existing files if needed **

    logging.info(f"Starting extraction for {len(data)} files...")

    """ Prepare Tasks """
    tasks = [
        process_file_worker(filename, text, config, frame_extractor)
        for filename, text in data
    ]

    successful_extractions = 0
    skipped_extractions = 0
    failed_extractions = 0

    """ Run Async Loop with Progress Bar """
    if tasks:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing files"):
            original_filename, status_message = await future
            
            if status_message == "success":
                successful_extractions += 1
            elif status_message == "skipped":
                skipped_extractions += 1
            else:
                failed_extractions += 1
                logging.warning(f"Processing failed for {original_filename}: {status_message}")

    logging.info("Extraction pipeline finished.")
    logging.info(f"Successfully processed: {successful_extractions} files.")
    logging.info(f"Skipped: {skipped_extractions} files.")
    logging.info(f"Failed to process: {failed_extractions} files.")
    logging.info(f"Output saved in: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())