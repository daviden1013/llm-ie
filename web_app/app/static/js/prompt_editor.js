// static/js/prompt_editor.js

function initializePromptEditor() {
    if (window.promptEditorInitialized) {
        return;
    }
    console.log("Running initializePromptEditor()...");

    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatButton = document.getElementById('clear-chat-btn');
    const llmApiSelect = document.getElementById('llm-api-select');
    const temperatureInput = document.getElementById('pe-temperature');
    const maxTokensInput = document.getElementById('pe-max-tokens');

    if (!chatHistory || !userInput || !sendButton || !clearChatButton || !llmApiSelect) {
        console.error("Prompt Editor: One or more critical UI elements not found. Initialization failed.");
        return;
    }

    let conversationHistory = [];
    let markedOptionsAreSet = false;
    const SCROLL_THRESHOLD = 20;

    function ensureMarkedOptions() {
        if (markedOptionsAreSet) return;
        if (typeof window.markedReady !== 'undefined' && window.markedReady && typeof marked !== 'undefined') {
            if (typeof window.hljsReady !== 'undefined' && window.hljsReady && typeof hljs !== 'undefined') {
                try {
                    marked.setOptions({
                        highlight: function(code, lang) {
                            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                            return hljs.highlight(code, { language, ignoreIllegals: true }).value;
                        },
                        pedantic: false, gfm: true, breaks: true, sanitize: false
                    });
                    markedOptionsAreSet = true;
                } catch (e) { console.error("Error setting Marked options with hljs:", e); }
            } else {
                try {
                    marked.setOptions({ pedantic: false, gfm: true, breaks: true, sanitize: false, highlight: null });
                    markedOptionsAreSet = true;
                } catch (e) { console.error("Error setting basic Marked options:", e); }
            }
        }
    }

    function formatMessageContent(text) {
        ensureMarkedOptions();
        if (typeof marked !== "undefined") {
            try {
                return marked.parse(text);
            } catch (e) {
                console.error("Error parsing markdown with Marked.js:", e);
            }
        }
        return text.replace(/&/g, "&amp;")
                   .replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;")
                   .replace(/\n/g, '<br>');
    }

    function savePromptEditorState() {
        localStorage.setItem('promptEditorChatHistory', JSON.stringify(conversationHistory));

        const controlsToSave = {
            llmApiSelectValue: llmApiSelect.value,
            temperatureValue: temperatureInput.value,
            maxTokensValue: maxTokensInput.value
        };

        document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
            if (el.id) {
                // MODIFIED: Exclude the removed checkboxes
                if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') {
                    return; 
                }
                if (el.type === 'checkbox') controlsToSave[el.id] = el.checked;
                else controlsToSave[el.id] = el.value;
            }
        });
        localStorage.setItem('promptEditorControls', JSON.stringify(controlsToSave));
    }

    function loadPromptEditorState() {
        const savedControls = JSON.parse(localStorage.getItem('promptEditorControls'));
        if (savedControls) {
            if (typeof savedControls.llmApiSelectValue !== 'undefined') llmApiSelect.value = savedControls.llmApiSelectValue;
            if (typeof savedControls.temperatureValue !== 'undefined') temperatureInput.value = savedControls.temperatureValue;
            if (typeof savedControls.maxTokensValue !== 'undefined') maxTokensInput.value = savedControls.maxTokensValue;

            document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
                if (el.id && typeof savedControls[el.id] !== 'undefined') {
                     // MODIFIED: Exclude the removed checkboxes
                    if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') {
                        return;
                    }
                    if (el.type === 'checkbox') el.checked = savedControls[el.id];
                    else el.value = savedControls[el.id];
                }
            });
        }
        if (typeof updateConditionalOptions === "function") {
            updateConditionalOptions();
        }

        const savedChat = JSON.parse(localStorage.getItem('promptEditorChatHistory'));
        if (savedChat && Array.isArray(savedChat)) {
            conversationHistory = savedChat;
            chatHistory.innerHTML = '';
            conversationHistory.forEach(message => {
                 _createAndAppendMessageDOM(message.role, message.content, false);
            });
            if (chatHistory.lastChild) {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        }
    }

    function _createAndAppendMessageDOM(role, rawText, shouldApplyConditionalScroll = true) {
        const messageContainerDiv = document.createElement('div');
        messageContainerDiv.classList.add(role === 'user' ? 'user-message' : 'assistant-message');
        
        let isNearBottom = false;
        if (chatHistory && shouldApplyConditionalScroll) {
            isNearBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + SCROLL_THRESHOLD;
        }

        ensureMarkedOptions();
        messageContainerDiv.innerHTML = formatMessageContent(rawText);

        if (role === 'assistant') {
            messageContainerDiv.dataset.rawText = rawText;
            const copyButton = document.createElement('button');
            copyButton.classList.add('copy-button', 'action-icon');
            copyButton.setAttribute('aria-label', 'Copy raw message text');
            copyButton.title = 'Copy Raw Text';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            messageContainerDiv.appendChild(copyButton);
        }

        chatHistory.appendChild(messageContainerDiv);

        if (typeof window.hljsReady !== 'undefined' && window.hljsReady && typeof hljs !== 'undefined' && markedOptionsAreSet) {
            messageContainerDiv.querySelectorAll('pre code').forEach((block) => {
                try { hljs.highlightElement(block); } catch (e) { console.error("Error highlighting element:", e, block); }
            });
        }
        
        if (chatHistory && shouldApplyConditionalScroll && isNearBottom) {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }
    
    function addMessageToChatDOM(role, text) {
        _createAndAppendMessageDOM(role, text, true);
    }

    function updateLastAssistantMessageDOM(currentRawMarkdown) {
        let lastMessageContainerDiv = chatHistory.querySelector('.assistant-message:last-child');
        
        if (!lastMessageContainerDiv) {
            addMessageToChatDOM('assistant', currentRawMarkdown);
            return;
        }

        let isNearBottom = false;
        if (chatHistory) {
            isNearBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + SCROLL_THRESHOLD;
        }

        let copyButton = lastMessageContainerDiv.querySelector('.copy-button');
        if (copyButton) {
            copyButton.remove();
        }
        
        ensureMarkedOptions();
        lastMessageContainerDiv.innerHTML = formatMessageContent(currentRawMarkdown);
        lastMessageContainerDiv.dataset.rawText = currentRawMarkdown;

        if (!copyButton) {
            copyButton = document.createElement('button');
            copyButton.classList.add('copy-button', 'action-icon');
            copyButton.setAttribute('aria-label', 'Copy raw message text');
            copyButton.title = 'Copy Raw Text';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        }
        lastMessageContainerDiv.appendChild(copyButton);

        if (typeof window.hljsReady !== 'undefined' && window.hljsReady && typeof hljs !== 'undefined' && markedOptionsAreSet) {
            lastMessageContainerDiv.querySelectorAll('pre code').forEach((block) => {
                try { hljs.highlightElement(block); } catch (e) { console.error("Error re-highlighting element:", e, block); }
            });
        }

        if (chatHistory && isNearBottom) {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }

    function updateConditionalOptions() {
        if (!llmApiSelect) return;
        const selectedApi = llmApiSelect.value;
        document.querySelectorAll('#llm-config-form .conditional-options').forEach(div => {
            div.style.display = 'none';
        });
        if (selectedApi) {
            const targetDivId = `${selectedApi}-options`;
            const optionsDiv = document.getElementById(targetDivId);
            if (optionsDiv) {
                optionsDiv.style.display = 'block';
            } else {
                console.warn(`updateConditionalOptions: Could not find options div for ID: ${targetDivId}`);
            }
        }
        if (selectedApi) hideApiSelectionWarning();
    }

    function showApiSelectionWarning() { if (llmApiSelect) llmApiSelect.classList.add('input-error'); }
    function hideApiSelectionWarning() { if (llmApiSelect) llmApiSelect.classList.remove('input-error'); }

    function getLlmConfiguration() {
        if (!llmApiSelect) {
            console.error("getLlmConfiguration: llmApiSelect element not found!");
            return { api_type: null };
        }
        const selectedApi = llmApiSelect.value;
        const config = {
            api_type: selectedApi,
            temperature: parseFloat(temperatureInput.value) || 0.2,
            max_tokens: parseInt(maxTokensInput.value) || 4096
        };

        if (selectedApi) {
            const optionsContainer = document.getElementById(`${selectedApi}-options`);
            if (optionsContainer) {
                optionsContainer.querySelectorAll('input, select, textarea').forEach(el => {
                    if (el.name && el.id) {
                        // MODIFIED: Exclude the removed checkboxes from being added to config
                        if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') {
                            return; 
                        }
                        let key = el.name || el.id;
                        if (el.type === 'checkbox') {
                            config[key] = el.checked;
                        } else if (el.type === 'number') {
                            config[key] = parseFloat(el.value) || (el.placeholder ? parseFloat(el.placeholder) : 0);
                        }
                         else {
                            config[key] = el.value;
                        }
                    }
                });
                 // Specific handling can be simplified as checkboxes are removed
                if (selectedApi === "openai_compatible") {
                    config.openai_compatible_api_key = document.getElementById('openai-compatible-api-key')?.value;
                    config.llm_base_url = document.getElementById('llm-base-url')?.value;
                    config.llm_model_openai_comp = document.getElementById('llm-model-openai-comp')?.value;
                } else if (selectedApi === "ollama") {
                    config.ollama_host = document.getElementById('ollama-host')?.value;
                    config.ollama_model = document.getElementById('ollama-model')?.value;
                    const numCtx = document.getElementById('ollama-num-ctx')?.value;
                    if (numCtx) config.ollama_num_ctx = parseInt(numCtx);
                } else if (selectedApi === "huggingface_hub") {
                    config.hf_token = document.getElementById('hf-token')?.value;
                    config.hf_model_or_endpoint = document.getElementById('hf-model-or-endpoint')?.value;
                } else if (selectedApi === "openai") {
                    config.openai_api_key = document.getElementById('openai-api-key')?.value;
                    config.openai_model = document.getElementById('openai-model')?.value;
                    // config.openai_reasoning_model = false; // MODIFIED: Default or remove if backend handles missing key
                } else if (selectedApi === "azure_openai") {
                    config.azure_openai_api_key = document.getElementById('azure-openai-api-key')?.value;
                    config.azure_endpoint = document.getElementById('azure-endpoint')?.value;
                    config.azure_api_version = document.getElementById('azure-api-version')?.value;
                    config.azure_deployment_name = document.getElementById('azure-deployment-name')?.value;
                    // config.azure_reasoning_model = false; // MODIFIED: Default or remove
                } else if (selectedApi === "litellm") {
                    config.litellm_model = document.getElementById('litellm-model')?.value;
                    config.litellm_api_key = document.getElementById('litellm-api-key')?.value;
                    config.litellm_base_url = document.getElementById('litellm-base-url')?.value;
                }
            }
        }
        return config;
    }

    function startChatStream(currentMessages, llmConfig) {
        userInput.disabled = true;
        sendButton.disabled = true;
        
        let assistantMessageContent = ""; 
        let firstChunkProcessed = false;
    
        fetch('/api/prompt-editor/chat', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           body: JSON.stringify({ messages: currentMessages, llmConfig: llmConfig })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || `HTTP error! Status: ${response.status}`);
                }).catch(() => {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                });
            }
            if (!response.body) {
                throw new Error("ReadableStream not available on the response.");
            }
            return response.body.getReader();
        })
        .then(reader => {
            const decoder = new TextDecoder();
            let sseBuffer = '';
    
            function processStream({ done, value }) {
                if (done) {
                    if (assistantMessageContent) {
                        const lastMsgIndex = conversationHistory.length - 1;
                        if (lastMsgIndex >= 0 && conversationHistory[lastMsgIndex].role === 'assistant') {
                            conversationHistory[lastMsgIndex].content = assistantMessageContent;
                        } else { 
                            conversationHistory.push({ role: 'assistant', content: assistantMessageContent });
                        }
                        savePromptEditorState();
                    }
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    userInput.focus();
                    return;
                }
    
                sseBuffer += decoder.decode(value, { stream: true });
                let lines = sseBuffer.split('\n');
                
                sseBuffer = lines.pop(); 
    
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try {
                            const jsonData = JSON.parse(line.substring(6));
                            if (jsonData.text) {
                                if (!firstChunkProcessed) {
                                    assistantMessageContent = jsonData.text;
                                    firstChunkProcessed = true;
                                } else {
                                    assistantMessageContent += jsonData.text;
                                }
                                updateLastAssistantMessageDOM(assistantMessageContent);
                            } else if (jsonData.error) {
                                console.error("Stream error from backend:", jsonData.error);
                                assistantMessageContent = `**Stream Error:** ${jsonData.error}`;
                                updateLastAssistantMessageDOM(assistantMessageContent);
                            }
                        } catch (e) {
                            // console.warn("Error parsing SSE JSON data:", e, "Original line:", line);
                        }
                    } else if (line.startsWith('event: end')) {
                        console.log("Stream indicated end via custom event.");
                    }
                });
    
                return reader.read().then(processStream);
            }
            return reader.read().then(processStream);
        })
        .catch(error => {
            console.error('Chat stream or fetch error:', error);
            updateLastAssistantMessageDOM(`**Error:** ${error.message}`);
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
            const lastMsgIndex = conversationHistory.length - 1;
            if (lastMsgIndex >= 0 && conversationHistory[lastMsgIndex].role === 'assistant' && 
                conversationHistory[lastMsgIndex].content === '...') {
                conversationHistory[lastMsgIndex].content = `**Error:** ${error.message}`;
                savePromptEditorState();
            }
        });
    }

    loadPromptEditorState();

    if (llmApiSelect) {
        llmApiSelect.addEventListener('change', () => {
            updateConditionalOptions();
            savePromptEditorState();
        });
    }

    const controlPanelElements = document.querySelectorAll(
        '#llm-config-form select, #llm-config-form input, #llm-config-form textarea'
    );
    controlPanelElements.forEach(element => {
        const eventType = (element.tagName.toLowerCase() === 'textarea' || 
                           (element.type && element.type.match(/text|url|password|number|search|email|tel/))) 
                          ? 'input' : 'change';
        element.addEventListener(eventType, savePromptEditorState);
    });

    if (sendButton) {
        sendButton.addEventListener('click', () => {
            const userText = userInput.value.trim();
            if (!userText) return;

            const llmConfig = getLlmConfiguration();
            if (!llmConfig.api_type) {
                showApiSelectionWarning();
                alert("Please select an LLM API from the controls panel.");
                return;
            }
            hideApiSelectionWarning();

            addMessageToChatDOM('user', userText);
            conversationHistory.push({ role: 'user', content: userText });
            userInput.value = '';

            addMessageToChatDOM('assistant', '...');
            conversationHistory.push({ role: 'assistant', content: '...' }); 
            
            savePromptEditorState();

            startChatStream(conversationHistory.slice(0, -1), llmConfig);
        });
    }

    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendButton.click();
            }
        });
    }

    if (clearChatButton) {
        clearChatButton.addEventListener('click', () => {
            if (chatHistory) chatHistory.innerHTML = '';
            conversationHistory = [];
            savePromptEditorState();
            userInput.disabled = false;
            sendButton.disabled = false;
            hideApiSelectionWarning();
        });
    }

    if (chatHistory) {
        chatHistory.addEventListener('click', function(event) {
            const copyButton = event.target.closest('.copy-button');
            if (copyButton) {
                const assistantMessageDiv = copyButton.closest('.assistant-message');
                if (assistantMessageDiv && typeof assistantMessageDiv.dataset.rawText !== 'undefined') {
                    const rawTextToCopy = assistantMessageDiv.dataset.rawText;
                    const originalIconHTML = copyButton.innerHTML;
                    const successIconHTML = '<i class="fas fa-check"></i>';
                    const errorIconHTML = '<i class="fas fa-times"></i>';

                    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                        navigator.clipboard.writeText(rawTextToCopy).then(() => {
                            copyButton.innerHTML = successIconHTML;
                            copyButton.classList.add('copied-success');
                            copyButton.disabled = true;
                            setTimeout(() => {
                                copyButton.innerHTML = originalIconHTML;
                                copyButton.classList.remove('copied-success');
                                copyButton.disabled = false;
                            }, 2000);
                        }).catch(err => {
                            console.error('Failed to copy text using navigator.clipboard: ', err);
                            copyButton.innerHTML = errorIconHTML;
                            copyButton.classList.add('copied-failed');
                            copyButton.disabled = true;
                            alert('Failed to copy text. Error: ' + err.message + '\nMake sure you are on a secure connection (HTTPS) or localhost, or check browser permissions.');
                            setTimeout(() => {
                                copyButton.innerHTML = originalIconHTML;
                                copyButton.classList.remove('copied-failed');
                                copyButton.disabled = false;
                            }, 3000);
                        });
                    } else {
                        console.warn('navigator.clipboard.writeText API not available. Trying legacy copy command.');
                        try {
                            const textArea = document.createElement("textarea");
                            textArea.value = rawTextToCopy;
                            textArea.style.position = "fixed";
                            textArea.style.top = "-9999px";
                            textArea.style.left = "-9999px";
                            document.body.appendChild(textArea);
                            textArea.focus();
                            textArea.select();
                            const successful = document.execCommand('copy');
                            document.body.removeChild(textArea);

                            if (successful) {
                                copyButton.innerHTML = successIconHTML;
                                copyButton.classList.add('copied-success');
                                copyButton.disabled = true;
                                setTimeout(() => {
                                    copyButton.innerHTML = originalIconHTML;
                                    copyButton.classList.remove('copied-success');
                                    copyButton.disabled = false;
                                }, 2000);
                            } else {
                                throw new Error('document.execCommand("copy") returned false.');
                            }
                        } catch (err) {
                            console.error('Legacy copy command failed: ', err);
                            copyButton.innerHTML = errorIconHTML;
                            copyButton.classList.add('copied-failed');
                            copyButton.disabled = true;
                            alert('Copying to clipboard is not supported or failed in this browser. Please copy manually.');
                            setTimeout(() => {
                                copyButton.innerHTML = originalIconHTML;
                                copyButton.classList.remove('copied-failed');
                                copyButton.disabled = false;
                            }, 3000);
                        }
                    }
                }
            }
        });
    }

    window.promptEditorInitialized = true;
    console.log("Prompt Editor UI and event listeners initialized.");
}