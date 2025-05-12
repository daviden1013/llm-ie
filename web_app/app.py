# app.py
import os
from flask import Flask, render_template, request, Response, jsonify, stream_with_context, current_app, send_file
import json
import time
import io # Keep for BytesIO in case other parts might use it, though not for doc.save()
import tempfile # Crucial for this fix and the new endpoint
from llm_ie.prompt_editor import PromptEditor
from llm_ie.extractors import DirectFrameExtractor
from utils.extractors import AppDirectFrameExtractor
from llm_ie.chunkers import (
    SentenceUnitChunker,
    WholeDocumentUnitChunker,
    NoContextChunker,
    WholeDocumentContextChunker,
    SlideWindowContextChunker
)
from llm_ie.engines import (
    InferenceEngine,
    OllamaInferenceEngine, OpenAIInferenceEngine, AzureOpenAIInferenceEngine,
    HuggingFaceHubInferenceEngine, LiteLLMInferenceEngine, LlamaCppInferenceEngine
)
from llm_ie.data_types import LLMInformationExtractionDocument, FrameExtractionUnitResult, LLMInformationExtractionFrame
import html
import traceback

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-very-secret-key-in-dev')

# --- LLM API Options ---
LLM_API_OPTIONS = [
    {"value": "openai_compatible", "name": "OpenAI Compatible (e.g., vLLM, Llama.cpp server)"},
    {"value": "ollama", "name": "Ollama"},
    {"value": "huggingface_hub", "name": "HuggingFace Hub/Endpoint"},
    {"value": "openai", "name": "OpenAI"},
    {"value": "azure_openai", "name": "Azure OpenAI"},
    {"value": "litellm", "name": "LiteLLM"},
]

# --- Helper Function to Create LLM Engine ---
def create_llm_engine(config: dict) -> InferenceEngine:
    api_type = config.get('api_type')
    if not api_type:
        raise ValueError("LLM API type ('api_type') is missing in the configuration.")
    app.logger.info(f"Attempting to create engine for API type: {api_type}")
    try:
        if api_type == "openai_compatible":
            base_url = config.get('llm_base_url')
            model = config.get('llm_model_openai_comp')
            api_key = config.get('openai_compatible_api_key', None)
            if not base_url or not model:
                 raise ValueError("Missing 'base_url' or 'model' for OpenAI Compatible.")
            return OpenAIInferenceEngine(model=model, api_key=api_key, base_url=base_url)
        elif api_type == "ollama":
            model = config.get('ollama_model')
            host = config.get('ollama_host') or None
            num_ctx = int(config.get('ollama_num_ctx', 4096))
            if not model:
                 raise ValueError("Missing 'model' for Ollama.")
            return OllamaInferenceEngine(model_name=model, host=host, num_ctx=num_ctx)
        elif api_type == "huggingface_hub":
            model_or_endpoint = config.get('hf_model_or_endpoint')
            token = config.get('hf_token') or None
            if not model_or_endpoint:
                raise ValueError("Missing 'Model Repo ID or Endpoint URL' for HuggingFace Hub.")
            return HuggingFaceHubInferenceEngine(model=model_or_endpoint, token=token)
        elif api_type == "openai":
            model = config.get('openai_model')
            api_key = config.get('openai_api_key') or None
            reasoning_model = config.get('openai_reasoning_model', False)
            if not model:
                 raise ValueError("Missing 'model' for OpenAI.")
            return OpenAIInferenceEngine(model=model, api_key=api_key, reasoning_model=reasoning_model)
        elif api_type == "azure_openai":
            deployment = config.get('azure_deployment_name')
            api_key = config.get('azure_openai_api_key') or None
            endpoint = config.get('azure_endpoint') or None
            api_version = config.get('azure_api_version') or None
            reasoning_model = config.get('azure_reasoning_model', False)
            if not deployment or not api_version:
                 raise ValueError("Missing 'deployment_name' or 'api_version' for Azure OpenAI.")
            return AzureOpenAIInferenceEngine(
                model=deployment, api_key=api_key, azure_endpoint=endpoint,
                api_version=api_version, reasoning_model=reasoning_model
            )
        elif api_type == "litellm":
            model_str = config.get('litellm_model')
            api_key = config.get('litellm_api_key') or None
            base_url = config.get('litellm_base_url') or None
            if not model_str:
                 raise ValueError("Missing 'model' string for LiteLLM.")
            return LiteLLMInferenceEngine(model=model_str, api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM API type: {api_type}")
    except KeyError as e:
         app.logger.error(f"Missing configuration key for {api_type}: {e}")
         raise ValueError(f"Missing configuration key for {api_type}: {e}") from e
    except ValueError as e:
        app.logger.error(f"Configuration error for {api_type}: {e}")
        raise e
    except ImportError as e:
        app.logger.error(f"Missing library for {api_type}: {e}")
        raise ImportError(f"Required library for {api_type} not installed: {e}") from e
    except Exception as e:
        app.logger.error(f"Failed to initialize engine for {api_type}: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize LLM engine for {api_type}.") from e

# --- Routes ---
@app.route('/')
def application_shell():
    return render_template('app_shell.html',
                           llm_api_options=LLM_API_OPTIONS,
                           active_tab='prompt-editor')

@app.route('/api/prompt-editor/chat', methods=['POST'])
def api_prompt_editor_chat():
    data = request.json
    messages = data.get('messages', [])
    llm_config = data.get('llmConfig', {})

    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    if not llm_config or not llm_config.get('api_type'):
         return jsonify({"error": "LLM API configuration ('api_type') is missing"}), 400

    try:
        engine_to_use = create_llm_engine(llm_config)
        app.logger.info(f"PromptEditor: Successfully created engine: {type(engine_to_use).__name__}")
        editor = PromptEditor(engine_to_use, DirectFrameExtractor)
        def generate():
            try:
                temperature = float(llm_config.get('temperature', 0.2))
                max_new_tokens = int(llm_config.get('max_tokens', 4096))
                engine_kwargs = {}
                stream = editor.chat_stream(messages=messages,
                                            temperature=temperature,
                                            max_new_tokens=max_new_tokens,
                                            **engine_kwargs)
                for chunk in stream:
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
            except Exception as e:
                current_app.logger.error(f"Error during stream generation: {e}\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'error': f'Stream generation failed: {type(e).__name__}'})}\n\n"
            finally:
                 yield "event: end\ndata: {}\n\n"
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    except (ValueError, ImportError, RuntimeError, TypeError) as e:
        app.logger.error(f"Failed to create LLM engine or process request: {e}")
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return jsonify({"error": f"{str(e)}"}), status_code
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/prompt-editor/chat: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500

def get_frame_extractor(engine: InferenceEngine, config: dict):
    prompt_template = config.get('prompt_template')
    if not prompt_template:
        raise ValueError("Prompt template is required for frame extractor.")
    extraction_unit_type = config.get('extraction_unit_type', 'Sentence')
    context_chunker_type = config.get('context_chunker_type', 'NoContext')
    if extraction_unit_type == "WholeDocument":
        unit_chunker = WholeDocumentUnitChunker()
    elif extraction_unit_type == "Sentence":
        unit_chunker = SentenceUnitChunker()
    elif extraction_unit_type == "TextLine":
        app.logger.warn("Mapping 'TextLine' extraction unit to SentenceUnitChunker.")
        unit_chunker = SentenceUnitChunker()
    else:
        raise ValueError(f"Unsupported extraction_unit_type: {extraction_unit_type}")
    if context_chunker_type == "NoContext":
        context_chunker = NoContextChunker()
    elif context_chunker_type == "WholeDocument":
        context_chunker = WholeDocumentContextChunker()
    elif context_chunker_type == "SlideWindow":
        window_size = int(config.get('slide_window_size', 2))
        context_chunker = SlideWindowContextChunker(window_size=window_size)
    else:
        raise ValueError(f"Unsupported context_chunker_type: {context_chunker_type}")
    app.logger.info(f"Instantiating AppDirectFrameExtractor with Unit Chunker: {type(unit_chunker).__name__} and Context Chunker: {type(context_chunker).__name__}")
    return AppDirectFrameExtractor(
        unit_chunker=unit_chunker,
        context_chunker=context_chunker,
        inference_engine=engine,
        prompt_template=prompt_template
    )

@app.route('/api/frame-extraction/stream', methods=['POST'])
def api_frame_extraction_stream():
    data = request.json
    input_text = data.get('inputText', '')
    llm_config = data.get('llmConfig', {})
    extractor_config_req = data.get('extractorConfig', {})

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    if not extractor_config_req.get('prompt_template'):
         return jsonify({"error": "Prompt template is required in extractorConfig"}), 400
    if not llm_config or not llm_config.get('api_type'):
         return jsonify({"error": "LLM API configuration ('api_type') is missing"}), 400

    try:
        engine_to_use = create_llm_engine(llm_config)
        app.logger.info(f"Frame Extraction: Created engine: {type(engine_to_use).__name__}")
        extractor = get_frame_extractor(engine_to_use, extractor_config_req)
        app.logger.info(f"Frame Extraction: Created extractor: {type(extractor).__name__}")
        def generate():
            all_extraction_unit_results = []
            try:
                stream_temperature = float(llm_config.get('temperature', 0.0))
                stream_max_tokens = int(llm_config.get('max_tokens', 512))
                app.logger.debug(f"Calling extractor.stream with temp={stream_temperature}, tokens={stream_max_tokens}")
                stream_generator = extractor.stream(
                    text_content=input_text,
                    temperature=stream_temperature,
                    max_new_tokens=stream_max_tokens
                )
                while True:
                    try:
                        event = next(stream_generator)
                        yield f"data: {json.dumps(event)}\n\n"
                    except StopIteration as e:
                        all_extraction_unit_results = e.value
                        if not isinstance(all_extraction_unit_results, list):
                            app.logger.warning(f"Extractor.stream() did not return a list. Got: {type(all_extraction_unit_results)}. Assuming empty.")
                            all_extraction_unit_results = []
                        break
                app.logger.info(f"Frame Extraction: Stream finished. Collected {len(all_extraction_unit_results)} unit results.")
                post_process_params = {
                    "case_sensitive": extractor_config_req.get('case_sensitive', False),
                    "fuzzy_match": extractor_config_req.get('fuzzy_match', True),
                    "allow_overlap_entities": extractor_config_req.get('allow_overlap_entities', False),
                }
                app.logger.debug(f"Calling extractor.post_process_frames with params: {post_process_params}")
                final_frames = extractor.post_process_frames(
                    extraction_results=all_extraction_unit_results,
                    **post_process_params
                )
                frames_dict = [f.to_dict() for f in final_frames]
                yield f"data: {json.dumps({'type': 'result', 'frames': frames_dict})}\n\n"
                app.logger.info(f"Frame Extraction: Post-processing complete, {len(final_frames)} frames found.")
            except Exception as e:
                current_app.logger.error(f"Error during frame extraction stream/processing: {e}\n{traceback.format_exc()}")
                error_payload = {'type': 'error', 'message': f'Extraction failed: {type(e).__name__} - {str(e)}'}
                yield f"data: {json.dumps(error_payload)}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n"
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    except (ValueError, ImportError, RuntimeError, TypeError) as e:
        app.logger.error(f"Failed to create engine/extractor or process request: {e}")
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return jsonify({"error": f"{str(e)}"}), status_code
    except Exception as e:
        app.logger.error(f"Unexpected error in /api/frame-extraction/stream setup: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during setup."}), 500

@app.route('/api/frame-extraction/download', methods=['POST'])
def api_frame_extraction_download():
    data = request.json
    input_text = data.get('inputText')
    frames_data_dicts = data.get('frames')

    if input_text is None or frames_data_dicts is None:
        return jsonify({"error": "Missing 'inputText' or 'frames' in request"}), 400

    tmp_file_path = None
    try:
        frame_objects = []
        if isinstance(frames_data_dicts, list):
            for frame_dict in frames_data_dicts:
                if not all(k in frame_dict for k in ['frame_id', 'start', 'end', 'entity_text']):
                    app.logger.warn(f"Skipping frame dict due to missing keys: {frame_dict}")
                    continue
                frame_objects.append(LLMInformationExtractionFrame.from_dict(frame_dict))

        doc = LLMInformationExtractionDocument(doc_id="downloaded_extraction", text=input_text)
        doc.add_frames(frame_objects, create_id=False)

        with tempfile.NamedTemporaryFile(suffix=".llmie", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name

        doc.save(tmp_file_path)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"extraction_results_{timestamp}.llmie"

        response = send_file(
            tmp_file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
        return response

    except Exception as e:
        app.logger.error(f"Error during download file preparation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to prepare download file: {str(e)}"}), 500
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                app.logger.info(f"Temporary file {tmp_file_path} deleted.")
            except Exception as e_remove:
                app.logger.error(f"Error deleting temporary file {tmp_file_path}: {e_remove}")


@app.route('/api/results/process_llmie_data', methods=['POST'])
def api_results_process_llmie_data():
    if 'llmie_file' not in request.files:
        app.logger.error("No 'llmie_file' part in the request.")
        return jsonify({"error": "No .llmie file part in the request"}), 400

    file_storage = request.files['llmie_file'] # Use a more descriptive name

    if file_storage.filename == '':
        app.logger.error("No file selected for upload.")
        return jsonify({"error": "No .llmie file selected"}), 400

    if not file_storage.filename.endswith('.llmie'):
        app.logger.error(f"Invalid file type uploaded: {file_storage.filename}")
        return jsonify({"error": "Invalid file type. Please upload an .llmie file"}), 400

    temp_file_path = None
    try:
        # Create a named temporary file, ensuring it's deleted
        # delete=False allows us to get the name and pass it, then delete manually
        with tempfile.NamedTemporaryFile(delete=False, suffix=".llmie", mode='wb') as tmp_file:
            file_storage.save(tmp_file) # Save the uploaded data to this temp file
            temp_file_path = tmp_file.name
        
        app.logger.info(f"Uploaded .llmie file saved temporarily to: {temp_file_path}")

        # Load the document using the temporary file path
        doc = LLMInformationExtractionDocument(filename=temp_file_path)
        app.logger.info(f"Successfully loaded .llmie document from temporary file: {doc.doc_id if doc.doc_id else 'No ID'}")

        # Extract data
        text_content = doc.text
        frames_list = [frame.to_dict() for frame in doc.frames]
        relations_list = [relation.to_dict() for relation in doc.relations] if doc.relations else []

        attribute_keys_set = set()
        if doc.frames:
            for frame in doc.frames: # Iterate over LLMInformationExtractionFrame objects
                if hasattr(frame, 'attr') and isinstance(frame.attr, dict):
                    for key in frame.attr.keys():
                        attribute_keys_set.add(key)
        
        sorted_attribute_keys = sorted(list(attribute_keys_set))
        app.logger.info(f"Extracted attribute keys: {sorted_attribute_keys}")

        return jsonify({
            "text": text_content,
            "frames": frames_list,
            "relations": relations_list,
            "attribute_keys": sorted_attribute_keys
        })

    except Exception as e:
        app.logger.error(f"Error processing .llmie file from temporary storage: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process .llmie file: {str(e)}"}), 500
    finally:
        # Robustly delete the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                app.logger.info(f"Temporary file {temp_file_path} deleted.")
            except Exception as e_remove:
                app.logger.error(f"Error deleting temporary file {temp_file_path}: {e_remove}")


# --- Endpoint for rendering visualization from provided data (remains as per previous correct version) ---
@app.route('/api/results/render', methods=['POST'])
def api_results_render():
    data = request.json
    text_content = data.get('text', "") 
    frames_data_dicts = data.get('frames', []) 
    relations_data_dicts = data.get('relations', [])
    viz_options = data.get('vizOptions', {})
    color_attr_key = viz_options.get('color_attr_key') or None

    try:
        app.logger.info(f"Rendering visualization with color_attr_key: {color_attr_key}")
        
        doc = LLMInformationExtractionDocument(doc_id="viz_doc_from_data", text=text_content)

        if frames_data_dicts:
            frames_to_add = []
            for frame_dict in frames_data_dicts:
                if all(k in frame_dict for k in ['frame_id', 'start', 'end', 'entity_text']):
                    frames_to_add.append(LLMInformationExtractionFrame.from_dict(frame_dict))
                else:
                    app.logger.warn(f"Skipping invalid frame dictionary for visualization: {frame_dict}")
            if frames_to_add:
                doc.add_frames(frames_to_add, create_id=False)
        
        if relations_data_dicts:
            doc.add_relations(relations_data_dicts)
        
        html_content = doc.viz_render(
            theme='light', 
            color_attr_key=color_attr_key
        )
        return jsonify({"html": html_content})

    except ImportError as ie:
         current_app.logger.error(f"Import Error in rendering: {ie}")
         return jsonify({"error": f"Visualization library not installed correctly: {ie}"}), 500
    except Exception as e:
        current_app.logger.error(f"Error in /api/results/render (from data): {e}", exc_info=True)
        return jsonify({"error": f"Failed to render visualization from provided data: {str(e)}"}), 500

# --- Run App ---
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO) # Changed to INFO for production, DEBUG can be too verbose
    app.logger.setLevel(logging.INFO)
    # For development, you might switch back to DEBUG:
    # logging.basicConfig(level=logging.DEBUG)
    # app.logger.setLevel(logging.DEBUG)
    app.run(debug=True, host='0.0.0.0', port=5001)