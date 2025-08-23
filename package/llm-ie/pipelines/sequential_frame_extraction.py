# This script processes many documents sequentially to extract frames
# Units in each document are concurrently processed using asyncio. Documents are sequentially in a single thread.
# ** Modify the config and logic as needed for your specific use case. **

import argparse
from easydict import EasyDict
import yaml
import os
import re
from tqdm import tqdm
import logging
from llm_ie import OpenAIInferenceEngine, DirectFrameExtractor, LLMInformationExtractionDocument, SentenceUnitChunker, SlideWindowContextChunker


def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("-c", "--config", help='dir to config file', type=str)
    add_arg("-o", "--overwrite", help="Overwrite existing output files", action="store_true")
    args = parser.parse_known_args()[0]
    
    """ Load config"""
    with open(args.config) as yaml_file:
        config = EasyDict(yaml.safe_load(yaml_file))

    """ Logging """
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('# Information extraction pipeline starts...')
        
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
    # Using .endswith for a slightly cleaner check
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

    """ Define engine """
    logging.info('Loading inference engine...')
    llm = OpenAIInferenceEngine(model="unused",
                                api_key="unused",
                                base_url=config["base_url"])

    """ Define extractor"""
    logging.info('Define extractor...')
    # Define unit chunker. Prompts sentences-by-sentence.
    unit_chunker = SentenceUnitChunker()
    # Define context chunker. Provides context for units.
    context_chunker = SlideWindowContextChunker(window_size=2)
    # Define extractor with the LLM, unit chunker, context chunker, prompt template, and system prompt.
    extractor = DirectFrameExtractor(llm, 
                                     unit_chunker=unit_chunker,
                                     context_chunker=context_chunker,
                                     prompt_template=prompt_template, 
                                     system_prompt=system_prompt)

    """ Process each file in the data directory """
    logging.info(f"Starting extraction for {len(data)} files...")
    loop = tqdm(data.items(), total=len(data), leave=True)
    for filename, text in loop:
        frames = extractor.extract_frames(text_content=text,
                                            case_sensitive=True,
                                            concurrent=True,
                                            concurrent_batch_size=64)
        
        llmie = LLMInformationExtractionDocument(doc_id=filename, text=text)
        llmie.add_frames(frames, create_id=True)
        llmie.save(os.path.join(config["out_dir"], config['run_name'], f"{llmie.doc_id}.llmie"))

if __name__ == "__main__":
    main()
    