# app/routes.py
# This file defines the web application routes.

import os
import json
import tempfile
import traceback
from flask import (
    Blueprint, render_template, request, Response, jsonify,
    stream_with_context, current_app, send_file
)

# Services from app_services.py
from .app_services import (
    create_llm_engine_from_config,
    get_app_frame_extractor
)

# Data types from your llm_ie library (if needed directly in routes, otherwise services handle them)
from llm_ie.data_types import LLMInformationExtractionDocument, LLMInformationExtractionFrame
from llm_ie.prompt_editor import PromptEditor
from llm_ie.engines import BasicLLMConfig, OpenAIReasoningLLMConfig, Qwen3LLMConfig
from llm_ie.extractors import DirectFrameExtractor # For PromptEditor type hint

# LLM API Options to pass to the template (could also be managed in app/__init__.py)
LLM_API_OPTIONS = [
    {"value": "openai_compatible", "name": "OpenAI Compatible"},
    {"value": "vllm", "name": "vLLM"},
    {"value": "ollama", "name": "Ollama"},
    {"value": "huggingface_hub", "name": "HuggingFace Hub"},
    {"value": "openai", "name": "OpenAI"},
    {"value": "azure_openai", "name": "Azure OpenAI"},
    {"value": "litellm", "name": "LiteLLM"},
]

# Create a Blueprint
main_bp = Blueprint('main', __name__, template_folder='templates', static_folder='static')

@main_bp.route('/')
def application_shell():
    """
    Serves the main application shell.
    """
    return render_template(
        'app_shell.html',
        llm_api_options=LLM_API_OPTIONS,
        active_tab='prompt-editor' # Default active tab
    )

@main_bp.route('/api/prompt-editor/chat', methods=['POST'])
def api_prompt_editor_chat():
    data = request.json
    messages = data.get('messages', [])
    llm_config_from_request = data.get('llmConfig', {})

    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    if not llm_config_from_request or not llm_config_from_request.get('api_type'):
        return jsonify({"error": "LLM API configuration ('api_type') is missing"}), 400
    
    try:
        # create_llm_engine_from_config now handles temperature and max_tokens via LLMConfig
        engine_to_use = create_llm_engine_from_config(llm_config_from_request)
        current_app.logger.info(f"PromptEditor: Successfully created engine: {type(engine_to_use).__name__}")
        # DirectFrameExtractor is a placeholder for the type expected by PromptEditor
        editor = PromptEditor(engine_to_use, DirectFrameExtractor) 

        def generate_chat_stream():
            try:
                stream = editor.chat_stream(
                    messages=messages
                )
                for chunk in stream:
                    yield f"data: {json.dumps(chunk)}\n\n"


            except Exception as e:
                current_app.logger.error(f"Error during PromptEditor stream generation: {e}\n{traceback.format_exc()}")
                yield f"data: {json.dumps({'error': f'Stream generation failed: {type(e).__name__} - {str(e)}'})}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n" 

        return Response(stream_with_context(generate_chat_stream()), mimetype='text/event-stream')

    except (ValueError, ImportError, RuntimeError, TypeError) as e:
        current_app.logger.error(f"Failed to create LLM engine or process PromptEditor request: {e}")
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return jsonify({"error": f"{str(e)}"}), status_code
    except Exception as e:
        current_app.logger.error(f"Unexpected error in /api/prompt-editor/chat: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred."}), 500


@main_bp.route('/api/frame-extraction/stream', methods=['POST'])
def api_frame_extraction_stream():
    """
    Handles requests for frame extraction.
    Streams the extraction process and results.
    """
    data = request.json
    input_text = data.get('inputText', '')
    llm_config_from_request = data.get('llmConfig', {})
    extractor_config_req = data.get('extractorConfig', {})

    if not input_text:
        return jsonify({"error": "Input text is required"}), 400
    if not extractor_config_req.get('prompt_template'):
        return jsonify({"error": "Prompt template is required in extractorConfig"}), 400
    if not llm_config_from_request or not llm_config_from_request.get('api_type'):
        return jsonify({"error": "LLM API configuration ('api_type') is missing"}), 400

    try:
        # create_llm_engine_from_config now handles temperature and max_tokens via LLMConfig
        engine_to_use = create_llm_engine_from_config(llm_config_from_request)
        current_app.logger.info(f"Frame Extraction: Created engine: {type(engine_to_use).__name__}")

        extractor = get_app_frame_extractor(engine_to_use, extractor_config_req)
        current_app.logger.info(f"Frame Extraction: Created extractor: {type(extractor).__name__}")

        def generate_extraction_stream():
            all_extraction_unit_results = []
            try:
                # Temperature and max_new_tokens are now part of the engine's config
                # qwen_no_think is also removed as it's not a standard InferenceEngine.chat param
                # If Qwen specific logic is needed, it should be in Qwen3LLMConfig.preprocess_messages

                current_app.logger.debug(f"Calling AppDirectFrameExtractor.stream...")
                # The `stream` method in `AppDirectFrameExtractor` itself calls `InferenceEngine.chat`
                # which now uses the LLMConfig for temperature and max_new_tokens.
                # So, we don't pass them here.
                stream_generator = extractor.stream(
                    text_content=input_text,
                    document_key=None # Assuming inputText is the direct document
                    # temperature, max_new_tokens, qwen_no_think removed from here
                )
                # ... (the rest of the stream processing logic for events remains the same) ...
                while True:
                    try:
                        event = next(stream_generator) # This will yield dicts like {"type": "info", "data": ...}
                        yield f"data: {json.dumps(event)}\n\n"
                    except StopIteration as e: 
                        all_extraction_unit_results = e.value # This is `collected_results` from your extractor's stream method
                        if not isinstance(all_extraction_unit_results, list):
                            current_app.logger.warning(f"Extractor.stream() did not return a list. Got: {type(all_extraction_unit_results)}. Assuming empty.")
                            all_extraction_unit_results = []
                        break 

                current_app.logger.info(f"Frame Extraction: Stream finished. Collected {len(all_extraction_unit_results)} unit results.")

                post_process_params = { # ... (remains the same)
                    "case_sensitive": extractor_config_req.get('case_sensitive', False),
                    "fuzzy_match": extractor_config_req.get('fuzzy_match', True),
                    "allow_overlap_entities": extractor_config_req.get('allow_overlap_entities', False),
                    "fuzzy_buffer_size": float(extractor_config_req.get('fuzzy_buffer_size', 0.2)),
                    "fuzzy_score_cutoff": float(extractor_config_req.get('fuzzy_score_cutoff', 0.8)),
                }
                current_app.logger.debug(f"Calling extractor.post_process_frames with params: {post_process_params}")

                final_frames = extractor.post_process_frames(
                    extraction_results=all_extraction_unit_results,
                    **post_process_params
                )
                frames_dict_list = [f.to_dict() for f in final_frames]
                yield f"data: {json.dumps({'type': 'result', 'frames': frames_dict_list})}\n\n"
                current_app.logger.info(f"Frame Extraction: Post-processing complete, {len(final_frames)} frames found.")

            except Exception as e: # ... (error handling in stream)
                current_app.logger.error(f"Error during frame extraction stream/processing: {e}\n{traceback.format_exc()}")
                error_payload = {'type': 'error', 'message': f'Extraction failed: {type(e).__name__} - {str(e)}'}
                yield f"data: {json.dumps(error_payload)}\n\n"
            finally:
                yield "event: end\ndata: {}\n\n" 

        return Response(stream_with_context(generate_extraction_stream()), mimetype='text/event-stream')

    except (ValueError, ImportError, RuntimeError, TypeError) as e:
        current_app.logger.error(f"Failed to create engine/extractor or process FrameExtraction request: {e}")
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return jsonify({"error": f"{str(e)}"}), status_code
    except Exception as e:
        current_app.logger.error(f"Unexpected error in /api/frame-extraction/stream setup: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during setup."}), 500


@main_bp.route('/api/frame-extraction/download', methods=['POST'])
def api_frame_extraction_download():
    """
    Handles requests to download extracted frames as an .llmie file.
    """
    data = request.json
    input_text = data.get('inputText')
    frames_data_dicts = data.get('frames') # Expects a list of frame dictionaries

    if input_text is None or frames_data_dicts is None:
        return jsonify({"error": "Missing 'inputText' or 'frames' in request"}), 400

    tmp_file_path = None
    try:
        frame_objects = []
        if isinstance(frames_data_dicts, list):
            for frame_dict in frames_data_dicts:
                # Basic validation for essential keys
                if not all(k in frame_dict for k in ['frame_id', 'start', 'end', 'entity_text']):
                    current_app.logger.warning(f"Skipping frame dict due to missing keys for download: {frame_dict}")
                    continue
                frame_objects.append(LLMInformationExtractionFrame.from_dict(frame_dict))

        # Create an LLMInformationExtractionDocument instance
        doc = LLMInformationExtractionDocument(doc_id="downloaded_extraction", text=input_text)
        doc.add_frames(frame_objects, create_id=False) # Assuming IDs are already present from client

        # Save to a temporary file
        # delete=False is important because send_file needs the file to exist after this block
        with tempfile.NamedTemporaryFile(suffix=".llmie", delete=False, mode='w', encoding='utf-8') as tmp_file:
            tmp_file_path = tmp_file.name
            doc.save(tmp_file_path) # LLMInformationExtractionDocument.save method

        timestamp = current_app.config.get("CURRENT_TIME", "timestamp") # Get time from app config or default
        filename = f"extraction_results_{timestamp}.llmie"

        # send_file will delete the temp file after sending if `tmp_file.delete=True` was used,
        # but since we used delete=False, we need to manage it in `finally`.
        # However, it's safer to let Flask handle it if possible, or clean up robustly.
        # For now, we rely on manual deletion in the finally block.
        return send_file(
            tmp_file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        current_app.logger.error(f"Error during download file preparation: {e}", exc_info=True)
        return jsonify({"error": f"Failed to prepare download file: {str(e)}"}), 500
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
                current_app.logger.info(f"Temporary download file {tmp_file_path} deleted.")
            except Exception as e_remove:
                current_app.logger.error(f"Error deleting temporary download file {tmp_file_path}: {e_remove}")


@main_bp.route('/api/results/process_llmie_data', methods=['POST'])
def api_results_process_llmie_data():
    """
    Processes an uploaded .llmie file and returns its content.
    """
    if 'llmie_file' not in request.files:
        current_app.logger.error("No 'llmie_file' part in the request.")
        return jsonify({"error": "No .llmie file part in the request"}), 400

    file_storage = request.files['llmie_file']

    if file_storage.filename == '':
        current_app.logger.error("No file selected for upload.")
        return jsonify({"error": "No .llmie file selected"}), 400

    if not file_storage.filename.endswith('.llmie'):
        current_app.logger.error(f"Invalid file type uploaded: {file_storage.filename}")
        return jsonify({"error": "Invalid file type. Please upload an .llmie file"}), 400

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".llmie") as tmp_file:
            file_storage.save(tmp_file) # Save the uploaded data to this temp file
            temp_file_path = tmp_file.name
        current_app.logger.info(f"Uploaded .llmie file saved temporarily to: {temp_file_path}")

        doc = LLMInformationExtractionDocument(filename=temp_file_path) # Load from the temp file
        current_app.logger.info(f"Successfully loaded .llmie document: {doc.doc_id if doc.doc_id else 'No ID'}")

        text_content = doc.text
        frames_list = [frame.to_dict() for frame in doc.frames]
        # Assuming relations in LLMInformationExtractionDocument are already dicts or have a to_dict()
        relations_list = doc.relations if doc.relations else []


        attribute_keys_set = set()
        if doc.frames:
            for frame in doc.frames:
                if hasattr(frame, 'attr') and isinstance(frame.attr, dict):
                    for key in frame.attr.keys():
                        attribute_keys_set.add(key)
        sorted_attribute_keys = sorted(list(attribute_keys_set))
        current_app.logger.info(f"Extracted attribute keys: {sorted_attribute_keys}")

        return jsonify({
            "text": text_content,
            "frames": frames_list,
            "relations": relations_list,
            "attribute_keys": sorted_attribute_keys
        })
    except Exception as e:
        current_app.logger.error(f"Error processing .llmie file: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process .llmie file: {str(e)}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                current_app.logger.info(f"Temporary .llmie file {temp_file_path} deleted.")
            except Exception as e_remove:
                current_app.logger.error(f"Error deleting temporary .llmie file {temp_file_path}: {e_remove}")


@main_bp.route('/api/results/render', methods=['POST'])
def api_results_render():
    """
    Renders visualization HTML from provided text, frames, and relations data.
    """
    data = request.json
    text_content = data.get('text', "")
    frames_data_dicts = data.get('frames', [])
    relations_data_dicts = data.get('relations', []) # Assuming this is already a list of dicts
    viz_options = data.get('vizOptions', {})
    color_attr_key = viz_options.get('color_attr_key') # Can be None or empty string

    try:
        if color_attr_key:
            current_app.logger.info(f"Rendering visualization with color_attr_key: '{color_attr_key}'")
        else:
            current_app.logger.info("Rendering visualization without color_attr_key")

        doc = LLMInformationExtractionDocument(doc_id="viz_doc_from_data", text=text_content)

        if frames_data_dicts:
            frames_to_add = []
            for frame_dict in frames_data_dicts:
                if all(k in frame_dict for k in ['frame_id', 'start', 'end', 'entity_text']):
                    frames_to_add.append(LLMInformationExtractionFrame.from_dict(frame_dict))
                else:
                    current_app.logger.warning(f"Skipping invalid frame dictionary for visualization: {frame_dict}")
            if frames_to_add:
                doc.add_frames(frames_to_add, create_id=False) # Assume IDs are final

        if relations_data_dicts: # relations_data_dicts should be a list of dicts
            doc.add_relations(relations_data_dicts)

        # Call viz_render from the LLMInformationExtractionDocument instance
        html_content = doc.viz_render(
            theme='light', # Or make this configurable
            color_attr_key=color_attr_key if color_attr_key else None # Pass None if empty string
        )
        return jsonify({"html": html_content})

    except ImportError as ie:
        current_app.logger.error(f"Import Error in rendering: {ie}")
        return jsonify({"error": f"Visualization library (e.g., ie-viz) not installed correctly: {ie}"}), 500
    except Exception as e:
        current_app.logger.error(f"Error in /api/results/render: {e}", exc_info=True)
        return jsonify({"error": f"Failed to render visualization: {str(e)}"}), 500

