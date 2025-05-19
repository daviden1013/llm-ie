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
    const llmConfigTypeSelect = document.getElementById('pe-llm-config-type-select'); // New
    const temperatureInput = document.getElementById('pe-temperature');
    const maxTokensInput = document.getElementById('pe-max-tokens');

    // Conditional Config Options elements for Prompt Editor
    const peOpenAIReasoningOptionsDiv = document.getElementById('pe-openai_reasoning-config-options');
    const peOpenAIReasoningEffortSelect = document.getElementById('pe-openai-reasoning-effort');
    const peQwen3OptionsDiv = document.getElementById('pe-qwen3-config-options');
    const peQwenThinkingModeCheckbox = document.getElementById('pe-qwen-thinking-mode');


    if (!chatHistory || !userInput || !sendButton || !clearChatButton || !llmApiSelect || !llmConfigTypeSelect) {
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

    function updateConditionalOptions() { // For LLM API selection
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

    function updateConditionalLLMConfigOptions() { // For LLM Config Type selection
        if (!llmConfigTypeSelect) return;
        const selectedConfigType = llmConfigTypeSelect.value;

        if (peOpenAIReasoningOptionsDiv) peOpenAIReasoningOptionsDiv.style.display = 'none';
        if (peQwen3OptionsDiv) peQwen3OptionsDiv.style.display = 'none';
        // Add more here if you have other conditional config divs for Prompt Editor

        if (selectedConfigType === 'OpenAIReasoningLLMConfig' && peOpenAIReasoningOptionsDiv) {
            peOpenAIReasoningOptionsDiv.style.display = 'block';
        } else if (selectedConfigType === 'Qwen3LLMConfig' && peQwen3OptionsDiv) {
            peQwen3OptionsDiv.style.display = 'block';
        }
        // Add more else-if for other types
    }


    function savePromptEditorState() {
        localStorage.setItem('promptEditorChatHistory', JSON.stringify(conversationHistory));

        const controlsToSave = {
            llmApiSelectValue: llmApiSelect.value,
            llmConfigTypeValue: llmConfigTypeSelect.value, // Save this
            temperatureValue: temperatureInput.value,
            maxTokensValue: maxTokensInput.value
        };

        // Save LLM API conditional options
        document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
            if (el.id) {
                // Ensure to skip the old reasoning model checkboxes
                if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') {
                    return;
                }
                if (el.type === 'checkbox') controlsToSave[el.id] = el.checked;
                else controlsToSave[el.id] = el.value;
            }
        });

        // Save LLM Config Type conditional options
        const selectedConfigType = llmConfigTypeSelect.value;
        if (selectedConfigType === 'OpenAIReasoningLLMConfig' && peOpenAIReasoningEffortSelect) {
            controlsToSave['pe-openai-reasoning-effort'] = peOpenAIReasoningEffortSelect.value;
        } else if (selectedConfigType === 'Qwen3LLMConfig' && peQwenThinkingModeCheckbox) {
            controlsToSave['pe-qwen-thinking-mode'] = peQwenThinkingModeCheckbox.checked;
        }
        // Add more for other config types

        localStorage.setItem('promptEditorControls', JSON.stringify(controlsToSave));
    }

    function loadPromptEditorState() {
        const savedControls = JSON.parse(localStorage.getItem('promptEditorControls'));
        if (savedControls) {
            if (typeof savedControls.llmApiSelectValue !== 'undefined') llmApiSelect.value = savedControls.llmApiSelectValue;
            if (typeof savedControls.llmConfigTypeValue !== 'undefined') { // Load this
                llmConfigTypeSelect.value = savedControls.llmConfigTypeValue;
            }
            if (typeof savedControls.temperatureValue !== 'undefined') temperatureInput.value = savedControls.temperatureValue;
            if (typeof savedControls.maxTokensValue !== 'undefined') maxTokensInput.value = savedControls.maxTokensValue;

            // Load LLM API conditional options
            document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
                if (el.id && typeof savedControls[el.id] !== 'undefined') {
                    if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') {
                        return;
                    }
                    if (el.type === 'checkbox') el.checked = savedControls[el.id];
                    else el.value = savedControls[el.id];
                }
            });
            
            // Load LLM Config Type conditional options
            const loadedConfigType = savedControls.llmConfigTypeValue || 'BasicLLMConfig';
            if (loadedConfigType === 'OpenAIReasoningLLMConfig' && peOpenAIReasoningEffortSelect && typeof savedControls['pe-openai-reasoning-effort'] !== 'undefined') {
                peOpenAIReasoningEffortSelect.value = savedControls['pe-openai-reasoning-effort'];
            } else if (loadedConfigType === 'Qwen3LLMConfig' && peQwenThinkingModeCheckbox && typeof savedControls['pe-qwen-thinking-mode'] !== 'undefined') {
                peQwenThinkingModeCheckbox.checked = savedControls['pe-qwen-thinking-mode'];
            }
            // Add more for other config types
        }

        updateConditionalOptions(); // For API specific options
        updateConditionalLLMConfigOptions(); // For LLM Config Type specific options


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

    function showApiSelectionWarning() { if (llmApiSelect) llmApiSelect.classList.add('input-error'); }
    function hideApiSelectionWarning() { if (llmApiSelect) llmApiSelect.classList.remove('input-error'); }

    function getLlmConfiguration() {
        if (!llmApiSelect) {
            console.error("getLlmConfiguration: llmApiSelect element not found!");
            return { api_type: null };
        }
        const selectedApi = llmApiSelect.value;
        const selectedLLMConfigType = llmConfigTypeSelect.value;

        const config = {
            api_type: selectedApi,
            llm_config_type: selectedLLMConfigType,
            temperature: parseFloat(temperatureInput.value) || 0.2,
            max_tokens: parseInt(maxTokensInput.value) || 4096
        };

        // Add parameters from the selected LLM API's conditional options
        if (selectedApi) {
            const optionsContainer = document.getElementById(`${selectedApi}-options`);
            if (optionsContainer) {
                optionsContainer.querySelectorAll('input, select, textarea').forEach(el => {
                    if (el.name && el.id) {
                        // Skip old reasoning checkboxes if they somehow still exist or are processed
                        if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') {
                            return; 
                        }
                        let key = el.name || el.id; // Prefer name, fallback to id
                        if (el.type === 'checkbox') {
                            config[key] = el.checked;
                        } else if (el.type === 'number') {
                            // For number inputs, try to parse as float, fallback to placeholder or 0
                            config[key] = parseFloat(el.value) || (el.placeholder ? parseFloat(el.placeholder) : (el.value === "0" ? 0 : null) );
                            if (config[key] === null && el.value !== "") config[key] = el.value; // If parsing failed but not empty, keep string
                        } else {
                            config[key] = el.value;
                        }
                    }
                });
            }
        }
        
        // Add parameters from the selected LLMConfigType's conditional options
        if (selectedLLMConfigType === 'OpenAIReasoningLLMConfig' && peOpenAIReasoningEffortSelect) {
            config.openai_reasoning_effort = peOpenAIReasoningEffortSelect.value;
        } else if (selectedLLMConfigType === 'Qwen3LLMConfig' && peQwenThinkingModeCheckbox) {
            config.qwen_thinking_mode = peQwenThinkingModeCheckbox.checked;
        }
        // Add more for other config types

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
                            let contentToUpdate = "";
                            // Adjust based on the actual structure yielded by the backend
                            // Assuming backend now sends: {'type': 'response'/'reasoning', 'data': 'text_chunk'}
                            // or still {'text': 'text_chunk'}
                            if (jsonData.text) { // If backend sends {'text': ...}
                                contentToUpdate = jsonData.text;
                            } else if (jsonData.data && jsonData.type === 'response') { // If backend sends {'type':'response', 'data':...}
                                contentToUpdate = jsonData.data;
                            } else if (jsonData.data && jsonData.type === 'reasoning') {
                                // Optionally display reasoning differently or prepend it
                                contentToUpdate = `*[${jsonData.type}]* ${jsonData.data} `;
                            } else if (jsonData.error) {
                                console.error("Stream error from backend:", jsonData.error);
                                contentToUpdate = `**Stream Error:** ${jsonData.error}`;
                            }

                            if (contentToUpdate) {
                                if (!firstChunkProcessed) {
                                    assistantMessageContent = contentToUpdate;
                                    firstChunkProcessed = true;
                                } else {
                                    assistantMessageContent += contentToUpdate;
                                }
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

    // Initial setup
    loadPromptEditorState(); // This now calls both updateConditionalOptions and updateConditionalLLMConfigOptions

    if (llmApiSelect) {
        llmApiSelect.addEventListener('change', () => {
            updateConditionalOptions();
            savePromptEditorState();
        });
    }
    if (llmConfigTypeSelect) { // Listener for the new select
        llmConfigTypeSelect.addEventListener('change', () => {
            updateConditionalLLMConfigOptions();
            savePromptEditorState(); // Save state when this changes too
        });
    }

    // Add event listeners to all control panel inputs for saving state
    const controlPanelElements = document.querySelectorAll(
        '#llm-config-form select, #llm-config-form input, #llm-config-form textarea'
    );
    controlPanelElements.forEach(element => {
        // Exclude the new LLM Config type specific inputs from this generic handler if they are already handled
        // or ensure their specific handlers also call savePromptEditorState.
        // For simplicity, we can let this generic handler also call savePromptEditorState,
        // it will just re-save but that's okay.
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