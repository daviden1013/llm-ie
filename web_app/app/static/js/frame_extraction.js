// static/js/frame_extraction.js
document.addEventListener('DOMContentLoaded', () => {
    // --- UI Element Selectors ---
    const inputTextElem = document.getElementById('fe-input-text');
    const promptTemplateElem = document.getElementById('fe-prompt-template');
    const extractionUnitElem = document.getElementById('fe-extraction-unit');
    const contextElem = document.getElementById('fe-context');
    const temperatureElem = document.getElementById('fe-temperature');
    const maxTokensElem = document.getElementById('fe-max-tokens');
    const fuzzyMatchElem = document.getElementById('fe-fuzzy-match');

    const startButton = document.getElementById('start-extraction-btn');
    const clearButton = document.getElementById('clear-extraction-btn');
    const downloadButton = document.getElementById('download-frames-btn');
    const outputElem = document.getElementById('extraction-output');
    const displayInputElem = document.getElementById('display-input-text');

    const feLlmApiSelect = document.getElementById('fe-llm-api-select');
    const feLlmConfigTypeSelect = document.getElementById('fe-llm-config-type-select');

    // Conditional LLM API Option Elements
    const feOpenaiCompatibleApiKey = document.getElementById('fe-openai-compatible-api-key');
    const feLlmBaseUrl = document.getElementById('fe-llm-base-url');
    const feLlmModelOpenaiComp = document.getElementById('fe-llm-model-openai-comp');
    const feOllamaHost = document.getElementById('fe-ollama-host');
    const feOllamaModel = document.getElementById('fe-ollama-model');
    const feOllamaNumCtx = document.getElementById('fe-ollama-num-ctx');
    const feHfToken = document.getElementById('fe-hf-token');
    const feHfModelOrEndpoint = document.getElementById('fe-hf-model-or-endpoint');
    const feOpenaiApiKey = document.getElementById('fe-openai-api-key');
    const feOpenaiModel = document.getElementById('fe-openai-model');
    // const feOpenaiReasoningModel = document.getElementById('fe-openai-reasoning-model'); // REMOVED
    const feAzureOpenaiApiKey = document.getElementById('fe-azure-openai-api-key');
    const feAzureEndpoint = document.getElementById('fe-azure-endpoint');
    const feAzureApiVersion = document.getElementById('fe-azure-api-version');
    const feAzureDeploymentName = document.getElementById('fe-azure-deployment-name');
    // const feAzureReasoningModel = document.getElementById('fe-azure-reasoning-model'); // REMOVED
    const feLitellmModel = document.getElementById('fe-litellm-model');
    const feLitellmApiKey = document.getElementById('fe-litellm-api-key');
    const feLitellmBaseUrl = document.getElementById('fe-litellm-base-url');
    
    const allFeApiOptionElements = [
        feOpenaiCompatibleApiKey, feLlmBaseUrl, feLlmModelOpenaiComp,
        feOllamaHost, feOllamaModel, feOllamaNumCtx,
        feHfToken, feHfModelOrEndpoint,
        feOpenaiApiKey, feOpenaiModel, // feOpenaiReasoningModel reference removed
        feAzureOpenaiApiKey, feAzureEndpoint, feAzureApiVersion, feAzureDeploymentName, // feAzureReasoningModel reference removed
        feLitellmModel, feLitellmApiKey, feLitellmBaseUrl
    ].filter(el => el); 

    // Conditional LLM Config Type Option Elements for Frame Extraction
    const feOpenAIReasoningOptionsDiv = document.getElementById('fe-openai_reasoning-config-options');
    const feOpenAIReasoningEffortSelect = document.getElementById('fe-openai-reasoning-effort');
    const feQwen3OptionsDiv = document.getElementById('fe-qwen3-config-options');
    const feQwenThinkingModeCheckbox = document.getElementById('fe-qwen-thinking-mode');

    let currentExtractedFrames = null; 

    // --- Helper Functions ---
    function escapeHTML(str) {
        if (typeof str !== 'string') {
            try {
                str = JSON.stringify(str, null, 2); 
            } catch (e) {
                str = String(str); 
            }
        }
        const div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    // --- LLM API Configuration UI Logic ---
    function feUpdateConditionalOptions() { 
        if (!feLlmApiSelect) return;
        const selectedApi = feLlmApiSelect.value;
        document.querySelectorAll('#fe-llm-config-form .conditional-options').forEach(div => {
            div.style.display = 'none';
        });
        if (selectedApi) {
            const targetDivId = `fe-${selectedApi}-options`;
            const optionsDiv = document.getElementById(targetDivId);
            if (optionsDiv) {
                optionsDiv.style.display = 'block';
            } else {
                console.warn(`FrameExtraction: updateConditionalOptions: Could not find div for ID: ${targetDivId}`);
            }
        }
        if (selectedApi) feHideApiSelectionWarning();
    }
    
    // --- LLM Config Type Configuration UI Logic ---
    function feUpdateConditionalLLMConfigOptions() { 
        if (!feLlmConfigTypeSelect) return;
        const selectedConfigType = feLlmConfigTypeSelect.value;

        if (feOpenAIReasoningOptionsDiv) feOpenAIReasoningOptionsDiv.style.display = 'none';
        if (feQwen3OptionsDiv) feQwen3OptionsDiv.style.display = 'none';

        if (selectedConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningOptionsDiv) {
            feOpenAIReasoningOptionsDiv.style.display = 'block';
        } else if (selectedConfigType === 'Qwen3LLMConfig' && feQwen3OptionsDiv) {
            feQwen3OptionsDiv.style.display = 'block';
        }
    }

    function feShowApiSelectionWarning() { if (feLlmApiSelect) feLlmApiSelect.classList.add('input-error'); }
    function feHideApiSelectionWarning() { if (feLlmApiSelect) feLlmApiSelect.classList.remove('input-error'); }

    function feGetLlmConfiguration() {
        if (!feLlmApiSelect) {
            console.error("FrameExtraction: feGetLlmConfiguration: feLlmApiSelect element not found!");
            return { api_type: null };
        }
        const selectedApi = feLlmApiSelect.value;
        const selectedLLMConfigType = feLlmConfigTypeSelect ? feLlmConfigTypeSelect.value : 'BasicLLMConfig';

        const config = {
            api_type: selectedApi,
            llm_config_type: selectedLLMConfigType,
            temperature: parseFloat(temperatureElem?.value) || 0.0,
            max_tokens: parseInt(maxTokensElem?.value) || 512
        };

        if (selectedApi && selectedApi !== "") {
            const optionsDivId = `fe-${selectedApi}-options`;
            const optionsDiv = document.getElementById(optionsDivId);
            if (optionsDiv) {
                optionsDiv.querySelectorAll('input, select, textarea').forEach(el => {
                    if (el.name && el.id) {
                        // The old reasoning model checkboxes (fe-openai-reasoning-model, fe-azure-reasoning-model)
                        // have been removed from HTML, so no specific check needed here for them.
                        // This loop will simply not find them.
                        let key = el.name.startsWith('fe_') ? el.name.substring(3) : el.name; 

                        if (el.type === 'checkbox') {
                            config[key] = el.checked;
                        } else if (el.type === 'number') {
                            const parsedValue = parseFloat(el.value);
                            config[key] = isNaN(parsedValue) ? (el.placeholder ? parseFloat(el.placeholder) : (el.value === "" ? null : el.value)) : parsedValue;
                        } else {
                            config[key] = el.value;
                        }
                    }
                });
            } else {
                console.warn(`FrameExtraction: feGetLlmConfiguration: Could not find options div for ID: ${optionsDivId}.`);
            }
        }
        
        if (selectedLLMConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningEffortSelect) {
            config.openai_reasoning_effort = feOpenAIReasoningEffortSelect.value;
        } else if (selectedLLMConfigType === 'Qwen3LLMConfig' && feQwenThinkingModeCheckbox) {
            config.qwen_thinking_mode = feQwenThinkingModeCheckbox.checked;
        }
        return config;
    }

    // --- State Management ---
    const feStatePrefix = 'frameExtraction_';
    function saveFrameExtractionState() {
        if (!localStorage) return;
        if (feLlmApiSelect) localStorage.setItem(`${feStatePrefix}llmApiSelectValue`, feLlmApiSelect.value);
        if (feLlmConfigTypeSelect) localStorage.setItem(`${feStatePrefix}llmConfigTypeValue`, feLlmConfigTypeSelect.value);

        allFeApiOptionElements.forEach(el => {
            if (el && el.id) { 
                 if (el.type === 'checkbox') localStorage.setItem(`${feStatePrefix}${el.id}`, el.checked);
                 else localStorage.setItem(`${feStatePrefix}${el.id}`, el.value);
            }
        });
        
        const selectedConfigType = feLlmConfigTypeSelect ? feLlmConfigTypeSelect.value : null;
        if (selectedConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningEffortSelect) {
            localStorage.setItem(`${feStatePrefix}fe-openai-reasoning-effort`, feOpenAIReasoningEffortSelect.value);
        } else if (selectedConfigType === 'Qwen3LLMConfig' && feQwenThinkingModeCheckbox) {
            localStorage.setItem(`${feStatePrefix}fe-qwen-thinking-mode`, feQwenThinkingModeCheckbox.checked);
        }

        if (inputTextElem) localStorage.setItem(`${feStatePrefix}inputText`, inputTextElem.value);
        if (promptTemplateElem) localStorage.setItem(`${feStatePrefix}promptTemplate`, promptTemplateElem.value);
        if (extractionUnitElem) localStorage.setItem(`${feStatePrefix}extractionUnit`, extractionUnitElem.value);
        if (contextElem) localStorage.setItem(`${feStatePrefix}contextType`, contextElem.value);
        if (temperatureElem) localStorage.setItem(`${feStatePrefix}temperature`, temperatureElem.value);
        if (maxTokensElem) localStorage.setItem(`${feStatePrefix}maxTokens`, maxTokensElem.value);
        if (fuzzyMatchElem) localStorage.setItem(`${feStatePrefix}fuzzyMatch`, fuzzyMatchElem.checked);

        if (outputElem) localStorage.setItem(`${feStatePrefix}extractionOutput`, outputElem.innerHTML);
        if (displayInputElem) localStorage.setItem(`${feStatePrefix}displayInputTextOutput`, displayInputElem.innerHTML);
    }

    function loadFrameExtractionState() {
        if (!localStorage) return;
        const savedApi = localStorage.getItem(`${feStatePrefix}llmApiSelectValue`);
        if (feLlmApiSelect && savedApi) {
            feLlmApiSelect.value = savedApi;
        }
        const savedLlmConfigType = localStorage.getItem(`${feStatePrefix}llmConfigTypeValue`); 
        if (feLlmConfigTypeSelect && savedLlmConfigType) {
            feLlmConfigTypeSelect.value = savedLlmConfigType;
        }
        
        allFeApiOptionElements.forEach(el => {
            if (el && el.id) { 
                const savedValue = localStorage.getItem(`${feStatePrefix}${el.id}`);
                if (savedValue !== null) {
                    if (el.type === 'checkbox') el.checked = (savedValue === 'true');
                    else el.value = savedValue;
                }
            }
        });
        
        const loadedConfigType = feLlmConfigTypeSelect ? feLlmConfigTypeSelect.value : 'BasicLLMConfig';
        if (loadedConfigType === 'OpenAIReasoningLLMConfig' && feOpenAIReasoningEffortSelect) {
            const savedEffort = localStorage.getItem(`${feStatePrefix}fe-openai-reasoning-effort`);
            if (savedEffort !== null) feOpenAIReasoningEffortSelect.value = savedEffort;
        } else if (loadedConfigType === 'Qwen3LLMConfig' && feQwenThinkingModeCheckbox) {
            const savedThinkingMode = localStorage.getItem(`${feStatePrefix}fe-qwen-thinking-mode`);
            if (savedThinkingMode !== null) feQwenThinkingModeCheckbox.checked = (savedThinkingMode === 'true');
        }

        const savedInputText = localStorage.getItem(`${feStatePrefix}inputText`);
        if (inputTextElem && savedInputText) inputTextElem.value = savedInputText;
        const savedPrompt = localStorage.getItem(`${feStatePrefix}promptTemplate`);
        if (promptTemplateElem && savedPrompt) promptTemplateElem.value = savedPrompt;
        
        const savedExtractionUnit = localStorage.getItem(`${feStatePrefix}extractionUnit`);
        if (extractionUnitElem && savedExtractionUnit) extractionUnitElem.value = savedExtractionUnit;
        const savedContextType = localStorage.getItem(`${feStatePrefix}contextType`);
        if (contextElem && savedContextType) contextElem.value = savedContextType;
        
        const savedTemp = localStorage.getItem(`${feStatePrefix}temperature`);
        if (temperatureElem && savedTemp) temperatureElem.value = savedTemp;
        const savedMaxTokens = localStorage.getItem(`${feStatePrefix}maxTokens`);
        if (maxTokensElem && savedMaxTokens) maxTokensElem.value = savedMaxTokens;
        const savedFuzzy = localStorage.getItem(`${feStatePrefix}fuzzyMatch`);
        if (fuzzyMatchElem && savedFuzzy !== null) fuzzyMatchElem.checked = (savedFuzzy === 'true');

        const savedExtractionOutput = localStorage.getItem(`${feStatePrefix}extractionOutput`);
        if (outputElem && savedExtractionOutput) {
            outputElem.innerHTML = savedExtractionOutput;
            const finalResultJsonMatch = savedExtractionOutput.match(/<pre class="final-result-json">(.*?)<\/pre>/s);
            if (finalResultJsonMatch && finalResultJsonMatch[1]) {
                try {
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = finalResultJsonMatch[1];
                    currentExtractedFrames = JSON.parse(tempDiv.textContent || tempDiv.innerText || "");
                    if (downloadButton && currentExtractedFrames && currentExtractedFrames.length > 0) {
                        downloadButton.disabled = false;
                    }
                } catch (e) {
                    console.warn("Could not parse frames from saved output on load:", e);
                    currentExtractedFrames = null;
                    if (downloadButton) downloadButton.disabled = true;
                }
            } else {
                 if (downloadButton) downloadButton.disabled = true;
            }
        } else {
            if (downloadButton) downloadButton.disabled = true;
        }
        const savedDisplayInputTextOutput = localStorage.getItem(`${feStatePrefix}displayInputTextOutput`);
        if (displayInputElem && savedDisplayInputTextOutput) {
            displayInputElem.innerHTML = savedDisplayInputTextOutput;
        }

        setTimeout(() => {
            feUpdateConditionalOptions(); 
            feUpdateConditionalLLMConfigOptions();
        }, 0);
    }

    // --- Event Listeners ---
    if (startButton) {
        startButton.addEventListener('click', () => {
            const llmConfig = feGetLlmConfiguration();
            if (!llmConfig.api_type) {
                feShowApiSelectionWarning();
                outputElem.innerHTML = '<div class="stream-error-message">Error: Please select an LLM API first.</div>';
                if (downloadButton) downloadButton.disabled = true; 
                currentExtractedFrames = null;
                saveFrameExtractionState(); 
                return;
            }
            feHideApiSelectionWarning();

            const inputText = inputTextElem.value;
            const promptTemplate = promptTemplateElem.value;
            const extractionUnit = extractionUnitElem.value;
            const contextType = contextElem.value;
            const fuzzyMatch = fuzzyMatchElem.checked;

            if (!inputText || !promptTemplate) {
                outputElem.innerHTML = '<div class="stream-error-message">Error: Input text and prompt template are required.</div>';
                if (downloadButton) downloadButton.disabled = true; 
                currentExtractedFrames = null;
                saveFrameExtractionState(); 
                return;
            }

            displayInputElem.innerHTML = escapeHTML(inputText);
            outputElem.innerHTML = 'Starting extraction...\n';
            startButton.disabled = true;
            clearButton.disabled = true;
            if (downloadButton) downloadButton.disabled = true; 
            currentExtractedFrames = null; 
            saveFrameExtractionState();

            let currentUnitId = null;
            let firstChunkForUnit = true;

            const payload = {
                llmConfig: llmConfig,
                inputText: inputText,
                extractorConfig: {
                    prompt_template: promptTemplate,
                    extraction_unit_type: extractionUnit, 
                    context_chunker_type: contextType,   
                    fuzzy_match: fuzzyMatch,
                    allow_overlap_entities: false, 
                    case_sensitive: false         
                }
            };

            fetch('/api/frame-extraction/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        let errorMsg = `HTTP error! status: ${response.status}`;
                        try {
                            const errJson = JSON.parse(text);
                            if (errJson && errJson.error) {
                                errorMsg += ` - ${errJson.error}`;
                            } else {
                                errorMsg += ` - ${text}`;
                            }
                        } catch (e) {
                            errorMsg += ` - ${text}`;
                        }
                        throw new Error(errorMsg);
                    });
                }
                if (!response.body) {
                    throw new Error("ReadableStream not available.");
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let finalFramesTemp = null; 

                function processText({ done, value }) {
                    if (done) {
                        if (finalFramesTemp) { 
                            currentExtractedFrames = finalFramesTemp; 
                            outputElem.innerHTML += `<hr><strong>Extracted Frames:</strong>\n<pre class="final-result-json">${escapeHTML(JSON.stringify(currentExtractedFrames, null, 2))}</pre>`;
                            if (downloadButton && currentExtractedFrames && currentExtractedFrames.length > 0) {
                                downloadButton.disabled = false; 
                            }
                        } else {
                            outputElem.innerHTML += '\n<hr><strong class="info-message">Extraction Finished. No frames extracted or result event not received.</strong>';
                            if (downloadButton) downloadButton.disabled = true;
                            currentExtractedFrames = null;
                        }
                        outputElem.scrollTop = outputElem.scrollHeight;
                        startButton.disabled = false;
                        clearButton.disabled = false;
                        currentUnitId = null;
                        saveFrameExtractionState(); 
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    let lines = buffer.split('\n');
                    buffer = lines.pop();
                    const oldScrollHeight = outputElem.scrollHeight;
                    const oldScrollTop = outputElem.scrollTop;
                    const clientHeight = outputElem.clientHeight;
                    const wasScrolledToBottom = oldScrollHeight - clientHeight <= oldScrollTop + 10;

                    lines.forEach(line => {
                        if (line.trim() === '') return;

                        if (line.startsWith('data: ')) {
                            const jsonDataString = line.substring(6);
                            if (jsonDataString.trim() === '{}' && lines.some(l => l.startsWith('event: end'))) {
                                return;
                            }
                            try {
                                const json = JSON.parse(jsonDataString);

                                if (outputElem.innerHTML === 'Starting extraction...\n') {
                                    outputElem.innerHTML = '';
                                }

                                switch (json.type) {
                                    case 'info':
                                        outputElem.innerHTML += `<div class="info-message"><strong>INFO:</strong> ${escapeHTML(json.data)}</div>`;
                                        break;
                                    case 'unit':
                                        currentUnitId = json.data.id;
                                        firstChunkForUnit = true;
                                        outputElem.innerHTML += `<hr><div class="unit-block" id="unit-block-${currentUnitId}">`;
                                        outputElem.innerHTML += `<h4 class="unit-header"><strong>UNIT [${escapeHTML(currentUnitId)}] (ID: ${escapeHTML(json.data.id)}, Range: ${escapeHTML(json.data.start)}-${escapeHTML(json.data.end)}):</strong></h4>`;
                                        outputElem.innerHTML += `<div class="unit-processed-text"><div class="text-snippet-header"><strong>Input Text for Unit:</strong></div><pre class="text-snippet-content">${escapeHTML(json.data.text)}</pre></div>`;
                                        outputElem.innerHTML += `<div class="unit-context-container" id="unit-context-${currentUnitId}"></div>`;
                                        outputElem.innerHTML += `<div class="unit-llm-output-container" id="unit-llm-output-${currentUnitId}"><span class="llm-output-header" style="display:none;"><strong>LLM Output:</strong></span><pre class="llm-output-content"></pre></div>`;
                                        outputElem.innerHTML += `</div>`;
                                        break;
                                    case 'context':
                                        if (currentUnitId !== null) {
                                            const contextContainer = document.getElementById(`unit-context-${currentUnitId}`);
                                            if (contextContainer) {
                                                contextContainer.innerHTML = `<div class="unit-context"><strong>Context Provided:</strong><pre>${escapeHTML(json.data)}</pre></div>`;
                                            }
                                        } else {
                                            outputElem.innerHTML += `<div class="general-context"><strong>CONTEXT:</strong><pre>${escapeHTML(json.data)}</pre></div>`;
                                        }
                                        break;
                                    case 'response': 
                                    case 'reasoning': 
                                        if (currentUnitId !== null) {
                                            const llmOutputContainer = document.getElementById(`unit-llm-output-${currentUnitId}`);
                                            if (llmOutputContainer) {
                                                const header = llmOutputContainer.querySelector('.llm-output-header');
                                                const content = llmOutputContainer.querySelector('.llm-output-content');
                                                
                                                if (firstChunkForUnit && header) { // Show header on first actual data chunk for the unit
                                                    header.style.display = 'inline';
                                                    firstChunkForUnit = false;
                                                }
                                                if (content && json.data) { 
                                                    if (json.type === 'reasoning') {
                                                        content.innerHTML += escapeHTML(`*[Reasoning]* ${json.data} `);
                                                    } else { // 'response'
                                                        content.innerHTML += escapeHTML(json.data);
                                                    }
                                                }
                                            }
                                        } else { 
                                            if (json.data) {
                                                if (json.type === 'reasoning') {
                                                    outputElem.innerHTML += escapeHTML(`*[Reasoning]* ${json.data} `);
                                                } else { // 'response'
                                                    outputElem.innerHTML += escapeHTML(json.data);
                                                }
                                            }
                                        }
                                        break;
                                    case 'result':
                                        finalFramesTemp = json.frames; 
                                        break;
                                    case 'error':
                                        outputElem.innerHTML += `<div class="stream-error-message"><strong>STREAM ERROR:</strong> ${escapeHTML(json.message)}</div>`;
                                        if (downloadButton) downloadButton.disabled = true;
                                        currentExtractedFrames = null;
                                        break;
                                    default:
                                        if (jsonDataString.trim() !== '{}') {
                                            console.warn("Received unhandled data type from backend stream:", json);
                                            outputElem.innerHTML += `<div class="unknown-message">UNHANDLED STREAM EVENT (${escapeHTML(json.type || 'undefined')}): ${escapeHTML(json.data !== undefined ? json.data : jsonDataString)}</div>`;
                                        }
                                }
                            } catch (e) {
                                console.error("Failed to parse SSE data:", e, "Line:", line);
                                if (jsonDataString.trim() !== '{}') {
                                    outputElem.innerHTML += `<div class="stream-error-message">Error parsing stream data. See console. (Line: ${escapeHTML(line)})</div>`;
                                }
                            }
                        } else if (line.startsWith('event: end')) {
                            console.log("Client: Received SSE 'event: end' signal.");
                        }
                    });
                    if (wasScrolledToBottom) {
                        outputElem.scrollTop = outputElem.scrollHeight;
                    }
                    reader.read().then(processText).catch(error => {
                        outputElem.innerHTML += `<div class="stream-error-message"><strong>Stream Reading Error:</strong> ${escapeHTML(error.toString())}</div>`;
                        startButton.disabled = false; clearButton.disabled = false; if (downloadButton) downloadButton.disabled = true; currentUnitId = null; currentExtractedFrames = null;
                        saveFrameExtractionState(); 
                    });
                }
                reader.read().then(processText).catch(initialReadError => {
                    outputElem.innerHTML = `<div class="stream-error-message">Error starting stream: ${escapeHTML(initialReadError.toString())}</div>`;
                    startButton.disabled = false; clearButton.disabled = false; if (downloadButton) downloadButton.disabled = true; currentUnitId = null; currentExtractedFrames = null;
                    saveFrameExtractionState();
                });
            })
            .catch(error => {
                outputElem.innerHTML = `<div class="stream-error-message">Error connecting to extraction API: ${escapeHTML(error.toString())}</div>`;
                startButton.disabled = false; clearButton.disabled = false; if (downloadButton) downloadButton.disabled = true; currentUnitId = null; currentExtractedFrames = null;
                saveFrameExtractionState();
            });
        });
    }

    if (clearButton) {
        clearButton.addEventListener('click', () => {
            outputElem.innerHTML = '';
            displayInputElem.innerHTML = ''; 
            startButton.disabled = false;
            clearButton.disabled = false;
            if (downloadButton) downloadButton.disabled = true; 
            currentExtractedFrames = null; 
            localStorage.removeItem(`${feStatePrefix}extractionOutput`);
            localStorage.removeItem(`${feStatePrefix}displayInputTextOutput`);
        });
    }

    if (downloadButton) {
        downloadButton.addEventListener('click', async () => {
            if (!currentExtractedFrames || currentExtractedFrames.length === 0) {
                alert("No frames available to download.");
                return;
            }
            const inputText = inputTextElem.value; 
            if (!inputText) {
                alert("Input text is missing, cannot create a complete .llmie file.");
                return;
            }

            downloadButton.disabled = true;
            downloadButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';

            try {
                const response = await fetch('/api/frame-extraction/download', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        inputText: inputText, 
                        frames: currentExtractedFrames
                    })
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const suggestedFilename = response.headers.get('Content-Disposition')?.split('filename=')[1]?.replace(/"/g, '') || 'extraction.llmie';
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = suggestedFilename;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } else {
                    const errorData = await response.json();
                    alert(`Download failed: ${errorData.error || response.statusText}`);
                }
            } catch (error) {
                console.error("Download error:", error);
                alert(`Download failed: ${error.message}`);
            } finally {
                downloadButton.disabled = false;
                downloadButton.innerHTML = '<i class="fas fa-download"></i>'; 
            }
        });
    }

    if (feLlmApiSelect) {
        feLlmApiSelect.addEventListener('change', () => {
            feUpdateConditionalOptions();
            saveFrameExtractionState();
        });
    }
    if (feLlmConfigTypeSelect) {
        feLlmConfigTypeSelect.addEventListener('change', () => {
            feUpdateConditionalLLMConfigOptions();
            saveFrameExtractionState();
        });
    }

    const feElementsToSaveOnInputOrChange = [
        inputTextElem, promptTemplateElem, 
        extractionUnitElem, contextElem, 
        temperatureElem, maxTokensElem, fuzzyMatchElem,
        ...allFeApiOptionElements,
        feLlmConfigTypeSelect, // The dropdown itself needs to trigger save on change
        feOpenAIReasoningEffortSelect,
        feQwenThinkingModeCheckbox
    ].filter(el => el); 

    feElementsToSaveOnInputOrChange.forEach(element => {
        if (element) { 
            const eventType = (element.tagName.toLowerCase() === 'textarea' || (element.type && element.type.match(/text|url|password|number|search|email|tel/))) ? 'input' : 'change';
            element.addEventListener(eventType, saveFrameExtractionState);
        }
    });

    loadFrameExtractionState(); 
});