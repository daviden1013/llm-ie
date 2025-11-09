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
    const llmConfigTypeSelect = document.getElementById('pe-llm-config-type-select');
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
    const SCROLL_THRESHOLD = 20; // Pixels from bottom to trigger auto-scroll
    
    // NEW: Add a variable to hold the AbortController for the current chat stream
    let currentChatStreamController = null;

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
        // Basic escaping if marked is not available
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
            llmConfigTypeValue: llmConfigTypeSelect.value, 
            temperatureValue: temperatureInput.value,
            maxTokensValue: maxTokensInput.value
        };

        document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
            if (el.id) {
                if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') return;
                if (el.type === 'checkbox') controlsToSave[el.id] = el.checked;
                else controlsToSave[el.id] = el.value;
            }
        });

        const selectedConfigType = llmConfigTypeSelect.value;
        if (selectedConfigType === 'OpenAIReasoningLLMConfig' && peOpenAIReasoningEffortSelect) {
            controlsToSave['pe-openai-reasoning-effort'] = peOpenAIReasoningEffortSelect.value;
        } else if (selectedConfigType === 'Qwen3LLMConfig' && peQwenThinkingModeCheckbox) {
            controlsToSave['pe-qwen-thinking-mode'] = peQwenThinkingModeCheckbox.checked;
        }
        localStorage.setItem('promptEditorControls', JSON.stringify(controlsToSave));
    }

    function loadPromptEditorState() {
        const savedControls = JSON.parse(localStorage.getItem('promptEditorControls'));
        if (savedControls) {
            if (typeof savedControls.llmApiSelectValue !== 'undefined') llmApiSelect.value = savedControls.llmApiSelectValue;
            if (typeof savedControls.llmConfigTypeValue !== 'undefined') {
                llmConfigTypeSelect.value = savedControls.llmConfigTypeValue;
            }
            if (typeof savedControls.temperatureValue !== 'undefined') temperatureInput.value = savedControls.temperatureValue;
            if (typeof savedControls.maxTokensValue !== 'undefined') maxTokensInput.value = savedControls.maxTokensValue;

            document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
                if (el.id && typeof savedControls[el.id] !== 'undefined') {
                    if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') return;
                    if (el.type === 'checkbox') el.checked = savedControls[el.id];
                    else el.value = savedControls[el.id];
                }
            });
            
            const loadedConfigType = savedControls.llmConfigTypeValue || 'BasicLLMConfig';
            if (loadedConfigType === 'OpenAIReasoningLLMConfig' && peOpenAIReasoningEffortSelect && typeof savedControls['pe-openai-reasoning-effort'] !== 'undefined') {
                peOpenAIReasoningEffortSelect.value = savedControls['pe-openai-reasoning-effort'];
            } else if (loadedConfigType === 'Qwen3LLMConfig' && peQwenThinkingModeCheckbox && typeof savedControls['pe-qwen-thinking-mode'] !== 'undefined') {
                peQwenThinkingModeCheckbox.checked = savedControls['pe-qwen-thinking-mode'];
            }
        }
        updateConditionalOptions(); 
        updateConditionalLLMConfigOptions(); 

        const savedChat = JSON.parse(localStorage.getItem('promptEditorChatHistory'));
        if (savedChat && Array.isArray(savedChat)) {
            conversationHistory = savedChat;
            chatHistory.innerHTML = ''; // Clear existing
            conversationHistory.forEach(message => {
                 _createAndAppendMessageDOM(message.role, message.content, false, message.reasoning || null);
            });
            if (chatHistory.lastChild) {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        }
    }

    function _createAndAppendMessageDOM(role, rawText, shouldApplyConditionalScroll = true, existingReasoning = null) {
        const messageContainerDiv = document.createElement('div');
        messageContainerDiv.classList.add(role === 'user' ? 'user-message' : 'assistant-message');
        
        let isNearBottom = false;
        if (chatHistory && shouldApplyConditionalScroll) {
            isNearBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + SCROLL_THRESHOLD;
        }
    
        ensureMarkedOptions(); 
    
        if (role === 'assistant') {
            const reasoningContainer = document.createElement('div');
            reasoningContainer.classList.add('reasoning-container');
            
            const detailsElement = document.createElement('details');
            const summaryElement = document.createElement('summary');
            summaryElement.innerHTML = 'Show Reasoning <span class="reasoning-icon">✨</span>';
            
            const reasoningTokensPre = document.createElement('pre');
            reasoningTokensPre.classList.add('reasoning-tokens');

            if (existingReasoning) {
                reasoningTokensPre.textContent = existingReasoning;
                reasoningContainer.style.display = 'block';
            } else {
                reasoningContainer.style.display = 'none'; 
            }
    
            detailsElement.appendChild(summaryElement);
            detailsElement.appendChild(reasoningTokensPre);
            reasoningContainer.appendChild(detailsElement);
            messageContainerDiv.appendChild(reasoningContainer);
    
            const messageContentDiv = document.createElement('div');
            messageContentDiv.classList.add('message-content');
            messageContentDiv.innerHTML = formatMessageContent(rawText); 
            messageContainerDiv.appendChild(messageContentDiv);
            
            messageContainerDiv.dataset.rawText = rawText;
    
            const copyButton = document.createElement('button');
            copyButton.classList.add('copy-button', 'action-icon');
            copyButton.setAttribute('aria-label', 'Copy main message text');
            copyButton.title = 'Copy Main Message Text';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            messageContentDiv.appendChild(copyButton); 
    
        } else { // User message
            messageContainerDiv.innerHTML = formatMessageContent(rawText);
        }
    
        chatHistory.appendChild(messageContainerDiv);
    
        if (typeof window.hljsReady !== 'undefined' && window.hljsReady && typeof hljs !== 'undefined' && markedOptionsAreSet) {
            messageContainerDiv.querySelectorAll('.message-content pre code').forEach((block) => { 
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

    function updateLastAssistantMessageDOM(chunkType, chunkData) {
        let lastMessageContainerDiv = chatHistory.querySelector('.assistant-message:last-child');
        
        if (!lastMessageContainerDiv) {
            // This block creates the initial assistant message with "..."
            _createAndAppendMessageDOM('assistant', '...', true); 
            lastMessageContainerDiv = chatHistory.querySelector('.assistant-message:last-child');
            if (!lastMessageContainerDiv) return; 

            const newMsgContentDiv = lastMessageContainerDiv.querySelector('.message-content');
            if (newMsgContentDiv) {
                // dataset.rawText will hold the accumulating raw response.
                // formatMessageContent will handle the "..." if rawText is indeed "..."
                lastMessageContainerDiv.dataset.rawText = "..."; 
                newMsgContentDiv.innerHTML = formatMessageContent("..."); // Render "..." via markdown
                newMsgContentDiv.dataset.isPlaceholder = 'true'; 

                let copyButton = newMsgContentDiv.querySelector('.copy-button');
                if (!copyButton) {
                    copyButton = document.createElement('button');
                    copyButton.classList.add('copy-button', 'action-icon');
                    copyButton.setAttribute('aria-label', 'Copy main message text');
                    copyButton.title = 'Copy Main Message Text';
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    newMsgContentDiv.appendChild(copyButton);
                }
            }
        }

        const reasoningContainer = lastMessageContainerDiv.querySelector('.reasoning-container');
        const reasoningTokensPre = lastMessageContainerDiv.querySelector('.reasoning-tokens');
        const messageContentDiv = lastMessageContainerDiv.querySelector('.message-content');
        let copyButton = messageContentDiv ? messageContentDiv.querySelector('.copy-button') : null;

        let isNearBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + SCROLL_THRESHOLD;

        if (chunkType === 'reasoning' && reasoningContainer && reasoningTokensPre) {
            if (reasoningContainer.style.display === 'none') {
                reasoningContainer.style.display = 'block';
            }
            reasoningTokensPre.textContent += chunkData; 
        } else if (chunkType === 'response' && messageContentDiv) {
            let currentRawMainText = lastMessageContainerDiv.dataset.rawText || "";
            
            // If it's the placeholder "...", replace it completely with the first chunk.
            // Otherwise, append.
            if (messageContentDiv.dataset.isPlaceholder === 'true' || currentRawMainText === "...") {
                currentRawMainText = chunkData; 
                messageContentDiv.dataset.isPlaceholder = 'false'; // Mark placeholder as handled
            } else {
                currentRawMainText += chunkData;
            }
            lastMessageContainerDiv.dataset.rawText = currentRawMainText;

            if (copyButton) copyButton.remove(); // Remove before re-rendering
            
            ensureMarkedOptions();
            messageContentDiv.innerHTML = formatMessageContent(currentRawMainText); // Re-render markdown

            // Re-append copy button
            if (!copyButton || !messageContentDiv.contains(copyButton)) {
                if (!copyButton) { 
                    copyButton = document.createElement('button');
                    copyButton.classList.add('copy-button', 'action-icon');
                    copyButton.setAttribute('aria-label', 'Copy main message text');
                    copyButton.title = 'Copy Main Message Text';
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                }
                messageContentDiv.appendChild(copyButton);
            }
            
            // Re-highlight if necessary
            if (typeof window.hljsReady !== 'undefined' && window.hljsReady && typeof hljs !== 'undefined' && markedOptionsAreSet) {
                messageContentDiv.querySelectorAll('pre code').forEach((block) => {
                    try { hljs.highlightElement(block); } catch (e) { console.error("Error re-highlighting element:", e, block); }
                });
            }
        } else if (chunkType === 'error' && messageContentDiv) {
            const errorText = `**Error:** ${chunkData}`;
            let currentRawMainText = lastMessageContainerDiv.dataset.rawText || "";
            if (messageContentDiv.dataset.isPlaceholder === 'true' || currentRawMainText === "...") {
                currentRawMainText = errorText;
                messageContentDiv.dataset.isPlaceholder = 'false';
            } else {
                currentRawMainText += `\n${errorText}`; 
            }
            lastMessageContainerDiv.dataset.rawText = currentRawMainText;

            if (copyButton) copyButton.remove();
            messageContentDiv.innerHTML = formatMessageContent(currentRawMainText); 
            if (copyButton) messageContentDiv.appendChild(copyButton);
        }

        if (isNearBottom) {
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

        if (selectedApi) {
            const optionsContainer = document.getElementById(`${selectedApi}-options`);
            if (optionsContainer) {
                optionsContainer.querySelectorAll('input, select, textarea').forEach(el => {
                    if (el.name && el.id) {
                        if (el.id === 'openai-reasoning-model' || el.id === 'azure-reasoning-model') return; 
                        let key = el.name || el.id; 
                        if (el.type === 'checkbox') {
                            config[key] = el.checked;
                        } else if (el.type === 'number') {
                            config[key] = parseFloat(el.value) || (el.placeholder ? parseFloat(el.placeholder) : (el.value === "0" ? 0 : null) );
                            if (config[key] === null && el.value !== "") config[key] = el.value; 
                        } else {
                            config[key] = el.value;
                        }
                    }
                });
            }
        }
        
        if (selectedLLMConfigType === 'OpenAIReasoningLLMConfig' && peOpenAIReasoningEffortSelect) {
            config.openai_reasoning_effort = peOpenAIReasoningEffortSelect.value;
        } else if (selectedLLMConfigType === 'Qwen3LLMConfig' && peQwenThinkingModeCheckbox) {
            config.qwen_thinking_mode = peQwenThinkingModeCheckbox.checked;
        }
        return config;
    }

    // MODIFIED: Renamed function and added controller parameter
    function startChatStream(currentMessages, llmConfig, controller) {
        userInput.disabled = true;
        // MODIFIED: Button is now handled by the sendButton click listener
        // sendButton.disabled = true; 
        
        fetch('/api/prompt-editor/chat', {
           method: 'POST',
           headers: { 'Content-Type': 'application/json' },
           // MODIFIED: Pass the AbortSignal to the fetch request
           signal: controller.signal, 
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
                    const lastMsgIndex = conversationHistory.length - 1;
                    if (lastMsgIndex >= 0 && conversationHistory[lastMsgIndex].role === 'assistant') {
                        const lastMessageDiv = chatHistory.querySelector('.assistant-message:last-child');
                        if (lastMessageDiv) {
                             const mainContentDiv = lastMessageDiv.querySelector('.message-content');
                             conversationHistory[lastMsgIndex].content = mainContentDiv ? (lastMessageDiv.dataset.rawText || mainContentDiv.textContent || "") : 'Error: Content missing';
                             
                             const reasoningTokensPre = lastMessageDiv.querySelector('.reasoning-tokens');
                             if (reasoningTokensPre && reasoningTokensPre.textContent) {
                                 conversationHistory[lastMsgIndex].reasoning = reasoningTokensPre.textContent;
                             }
                        }
                    }
                    savePromptEditorState();
                    
                    // MODIFIED: Reset UI and controller
                    resetChatUI();
                    
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
                            
                            if (jsonData && jsonData.type && jsonData.data !== undefined) {
                                updateLastAssistantMessageDOM(jsonData.type, jsonData.data);
                            } else if (jsonData.text) { // Fallback for old {'text': chunk}
                                updateLastAssistantMessageDOM('response', jsonData.text);
                            } else if (jsonData.error) {
                                console.error("Stream error from backend:", jsonData.error);
                                updateLastAssistantMessageDOM('error', jsonData.error);
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
            // MODIFIED: Handle AbortError specifically
            if (error.name === 'AbortError') {
                console.log('Chat stream aborted by user.');
                updateLastAssistantMessageDOM('error', '[Generation stopped by user]');
                
                // Save the incomplete message (with the stop notice) to history
                const lastMsgIndex = conversationHistory.length - 1;
                if (lastMsgIndex >= 0 && conversationHistory[lastMsgIndex].role === 'assistant') {
                    const lastMessageDiv = chatHistory.querySelector('.assistant-message:last-child');
                    if (lastMessageDiv) {
                         const mainContentDiv = lastMessageDiv.querySelector('.message-content');
                         conversationHistory[lastMsgIndex].content = mainContentDiv ? (lastMessageDiv.dataset.rawText || mainContentDiv.textContent || "") : '[Generation stopped by user]';
                         
                         const reasoningTokensPre = lastMessageDiv.querySelector('.reasoning-tokens');
                         if (reasoningTokensPre && reasoningTokensPre.textContent) {
                             conversationHistory[lastMsgIndex].reasoning = reasoningTokensPre.textContent;
                         }
                    }
                    savePromptEditorState();
                }
            } else {
                console.error('Chat stream or fetch error:', error);
                updateLastAssistantMessageDOM('error', error.message); // Show error in the message area
                
                // Update conversationHistory with the error
                const lastMsgIndex = conversationHistory.length - 1;
                if (lastMsgIndex >= 0 && conversationHistory[lastMsgIndex].role === 'assistant' && 
                    (conversationHistory[lastMsgIndex].content === '...' || conversationHistory[lastMsgIndex].content === '')) {
                    conversationHistory[lastMsgIndex].content = `**Error:** ${error.message}`;
                    savePromptEditorState();
                }
            }

            // MODIFIED: Reset UI and controller in all error/abort cases
            resetChatUI();
            userInput.focus();
        });
    }

    // NEW: Helper function to reset the chat UI state
    function resetChatUI() {
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>'; // MODIFIED: Use icon
        sendButton.title = 'Send message'; // MODIFIED: Update title
        sendButton.classList.remove('stop-button');
        currentChatStreamController = null; // Clear the controller
    }


    // Initial setup
    loadPromptEditorState(); 

    if (llmApiSelect) {
        llmApiSelect.addEventListener('change', () => {
            updateConditionalOptions();
            savePromptEditorState();
        });
    }
    if (llmConfigTypeSelect) { 
        llmConfigTypeSelect.addEventListener('change', () => {
            updateConditionalLLMConfigOptions();
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
        // MODIFIED: Updated sendButton click listener
        sendButton.addEventListener('click', () => {
            
            // Check if a stream is currently active
            if (currentChatStreamController) {
                // If active, abort the stream
                currentChatStreamController.abort();
                // resetChatUI() will be called from the fetch's catch block
                return;
            }

            // If no stream is active, send a new message
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
            const lastAssistantMessageElement = chatHistory.querySelector('.assistant-message:last-child .message-content');
            if(lastAssistantMessageElement) {
                lastAssistantMessageElement.dataset.isPlaceholder = 'true'; 
            }
            conversationHistory.push({ role: 'assistant', content: '...' }); 
            
            savePromptEditorState();

            // NEW: Create a new AbortController for this request
            currentChatStreamController = new AbortController();
            
            // NEW: Change button to "Stop"
            sendButton.innerHTML = '<i class="fas fa-stop"></i>'; // MODIFIED: Use icon
            sendButton.title = 'Stop generation'; // MODIFIED: Update title
            sendButton.classList.add('stop-button');
            sendButton.disabled = false; // Ensure it's enabled to be clicked as "Stop"
            userInput.disabled = true; // Keep textarea disabled

            // MODIFIED: Pass the controller to the stream function
            startChatStream(conversationHistory.slice(0, -1), llmConfig, currentChatStreamController);
        });
    }

    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                // Trigger click only if a stream is NOT active
                if (!currentChatStreamController) {
                    sendButton.click();
                }
            }
        });
    }

    if (clearChatButton) {
        clearChatButton.addEventListener('click', () => {
            // NEW: If a stream is active, stop it before clearing
            if (currentChatStreamController) {
                currentChatStreamController.abort();
            }

            if (chatHistory) chatHistory.innerHTML = '';
            conversationHistory = [];
            savePromptEditorState(); // Save empty state
            
            // MODIFIED: Use resetChatUI to ensure button is correct
            resetChatUI(); 
            
            hideApiSelectionWarning();
        });
    }

    if (chatHistory) {
        chatHistory.addEventListener('click', function(event) {
            const copyButton = event.target.closest('.copy-button');
            if (copyButton) {
                const messageContentDiv = copyButton.closest('.message-content');
                const assistantMessageDiv = messageContentDiv ? messageContentDiv.closest('.assistant-message') : null;

                if (assistantMessageDiv && messageContentDiv && typeof assistantMessageDiv.dataset.rawText !== 'undefined') {
                    const rawTextToCopy = assistantMessageDiv.dataset.rawText;
                    const originalIconHTML = copyButton.innerHTML;
                    const successIconHTML = '<i class="fas fa-check"></i>';
                    const errorIconHTML = '<i class="fas fa-times"></i>';

                    // Clipboard API logic (same as your existing code)
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
                            // ... (rest of error handling)
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
                        // Legacy copy command (same as your existing code)
                        console.warn('navigator.clipboard.writeText API not available. Trying legacy copy command.');
                        try {
                            const textArea = document.createElement("textarea");
                            textArea.value = rawTextToCopy;
                            textArea.style.position = "fixed"; // Prevent scrolling to bottom
                            textArea.style.top = "-9999px";
                            textArea.style.left = "-9999px";
                            document.body.appendChild(textArea);
                            textArea.focus();
                            textArea.select();
                            const successful = document.execCommand('copy');
                            document.body.removeChild(textArea);

                            if (successful) {
                                copyButton.innerHTML = successIconHTML;
                                // ... (rest of success handling)
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
                            // ... (rest of error handling)
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