// static/js/prompt_editor.js

// This flag is set by app_shell.html once this script's main function is called
// window.promptEditorInitialized = window.promptEditorInitialized || false; // Managed in app_shell.html

function initializePromptEditor() {
    // Prevent re-initialization
    if (window.promptEditorInitialized) {
        // console.log("Prompt Editor JS already initialized. Skipping.");
        return;
    }
    console.log("Running initializePromptEditor()...");

    // --- UI Element Getters ---
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const clearChatButton = document.getElementById('clear-chat-btn');
    const llmApiSelect = document.getElementById('llm-api-select');
    const temperatureInput = document.getElementById('pe-temperature');
    const maxTokensInput = document.getElementById('pe-max-tokens');
    // Add other control panel elements if they need direct manipulation beyond save/load

    if (!chatHistory || !userInput || !sendButton || !clearChatButton || !llmApiSelect) {
        console.error("Prompt Editor: One or more critical UI elements not found. Initialization failed.");
        return;
    }

    // --- State Variables ---
    let conversationHistory = []; // Holds the actual conversation data {role, content}
    let markedOptionsAreSet = false;
    const SCROLL_THRESHOLD = 20; // Pixels from bottom to consider "at bottom" for auto-scroll

    // --- Markdown & Highlighting Setup ---
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
                    // console.log("Marked.js options set with highlight.js.");
                } catch (e) { console.error("Error setting Marked options with hljs:", e); }
            } else {
                // console.warn("Marked.js ready, but Highlight.js not. Setting basic Marked options.");
                try {
                    marked.setOptions({ pedantic: false, gfm: true, breaks: true, sanitize: false, highlight: null });
                    markedOptionsAreSet = true;
                } catch (e) { console.error("Error setting basic Marked options:", e); }
            }
        } else {
            // console.warn("Marked.js not yet ready for option setting.");
        }
    }

    function formatMessageContent(text) {
        ensureMarkedOptions();
        if (typeof marked !== "undefined") {
            try {
                return marked.parse(text); // marked.parse() is the correct modern API call
            } catch (e) {
                console.error("Error parsing markdown with Marked.js:", e);
            }
        }
        // Basic fallback if Marked.js fails or isn't loaded (should be rare)
        return text.replace(/&/g, "&amp;")
                   .replace(/</g, "&lt;")
                   .replace(/>/g, "&gt;")
                   .replace(/\n/g, '<br>');
    }

    // --- State Management (LocalStorage) ---
    function savePromptEditorState() {
        localStorage.setItem('promptEditorChatHistory', JSON.stringify(conversationHistory));

        const controlsToSave = {
            llmApiSelectValue: llmApiSelect.value,
            temperatureValue: temperatureInput.value,
            maxTokensValue: maxTokensInput.value
        };

        document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
            if (el.id) {
                if (el.type === 'checkbox') controlsToSave[el.id] = el.checked;
                else controlsToSave[el.id] = el.value;
            }
        });
        localStorage.setItem('promptEditorControls', JSON.stringify(controlsToSave));
        // console.log("Prompt Editor State Saved");
    }

    function loadPromptEditorState() {
        const savedControls = JSON.parse(localStorage.getItem('promptEditorControls'));
        if (savedControls) {
            if (typeof savedControls.llmApiSelectValue !== 'undefined') llmApiSelect.value = savedControls.llmApiSelectValue;
            if (typeof savedControls.temperatureValue !== 'undefined') temperatureInput.value = savedControls.temperatureValue;
            if (typeof savedControls.maxTokensValue !== 'undefined') maxTokensInput.value = savedControls.maxTokensValue;

            document.querySelectorAll('#llm-config-form .conditional-options input, #llm-config-form .conditional-options select, #llm-config-form .conditional-options textarea').forEach(el => {
                if (el.id && typeof savedControls[el.id] !== 'undefined') {
                    if (el.type === 'checkbox') el.checked = savedControls[el.id];
                    else el.value = savedControls[el.id];
                }
            });
        }
        // IMPORTANT: Update conditional options display after loading values
        if (typeof updateConditionalOptions === "function") {
            updateConditionalOptions(); // Ensure this function exists and is correctly scoped or globally available
        }

        const savedChat = JSON.parse(localStorage.getItem('promptEditorChatHistory'));
        if (savedChat && Array.isArray(savedChat)) {
            conversationHistory = savedChat;
            chatHistory.innerHTML = ''; // Clear current display
            conversationHistory.forEach(message => {
                // Render messages without trying to auto-scroll each one during bulk load
                 _createAndAppendMessageDOM(message.role, message.content, false);
            });
            if (chatHistory.lastChild) { // After all messages are loaded, scroll to the bottom
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        }
        // console.log("Prompt Editor State Loaded");
    }

    // --- Chat Message DOM Manipulation & Scrolling ---
    function _createAndAppendMessageDOM(role, rawText, shouldApplyConditionalScroll = true) {
        const messageContainerDiv = document.createElement('div');
        messageContainerDiv.classList.add(role === 'user' ? 'user-message' : 'assistant-message');
        
        let isNearBottom = false;
        if (chatHistory && shouldApplyConditionalScroll) {
            isNearBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + SCROLL_THRESHOLD;
        }

        ensureMarkedOptions(); // Ensure marked.js is ready
        messageContainerDiv.innerHTML = formatMessageContent(rawText); // Use rawText for Marked.js

        if (role === 'assistant') {
            messageContainerDiv.dataset.rawText = rawText; // Store raw markdown for copy functionality
            const copyButton = document.createElement('button');
            copyButton.classList.add('copy-button', 'action-icon');
            copyButton.setAttribute('aria-label', 'Copy raw message text');
            copyButton.title = 'Copy Raw Text';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
            messageContainerDiv.appendChild(copyButton);
        }

        chatHistory.appendChild(messageContainerDiv);

        // Apply syntax highlighting to new code blocks
        if (typeof window.hljsReady !== 'undefined' && window.hljsReady && typeof hljs !== 'undefined' && markedOptionsAreSet) {
            messageContainerDiv.querySelectorAll('pre code').forEach((block) => {
                try { hljs.highlightElement(block); } catch (e) { console.error("Error highlighting element:", e, block); }
            });
        }
        
        // Conditional scroll after DOM update
        if (chatHistory && shouldApplyConditionalScroll && isNearBottom) {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }
    
    // Called for new messages added by user or initial '...' from assistant
    function addMessageToChatDOM(role, text) {
        _createAndAppendMessageDOM(role, text, true); // Enable conditional scroll
    }

    // Called to update the streaming assistant message
    function updateLastAssistantMessageDOM(currentRawMarkdown) {
        let lastMessageContainerDiv = chatHistory.querySelector('.assistant-message:last-child');
        
        if (!lastMessageContainerDiv) { // Should not happen if placeholder was added
            addMessageToChatDOM('assistant', currentRawMarkdown);
            return;
        }

        let isNearBottom = false;
        if (chatHistory) {
            isNearBottom = chatHistory.scrollHeight - chatHistory.clientHeight <= chatHistory.scrollTop + SCROLL_THRESHOLD;
        }

        let copyButton = lastMessageContainerDiv.querySelector('.copy-button');
        if (copyButton) { // Detach copy button before re-rendering content
            copyButton.remove();
        }
        
        ensureMarkedOptions();
        lastMessageContainerDiv.innerHTML = formatMessageContent(currentRawMarkdown);
        lastMessageContainerDiv.dataset.rawText = currentRawMarkdown; // Update raw text for copy

        // Re-attach or create the copy button
        if (!copyButton) { // Should ideally always find one
            copyButton = document.createElement('button');
            copyButton.classList.add('copy-button', 'action-icon');
            // ... (set attributes for copyButton as in _createAndAppendMessageDOM) ...
            copyButton.setAttribute('aria-label', 'Copy raw message text');
            copyButton.title = 'Copy Raw Text';
            copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        }
        lastMessageContainerDiv.appendChild(copyButton);

        // Re-apply syntax highlighting
        if (typeof window.hljsReady !== 'undefined' && window.hljsReady && typeof hljs !== 'undefined' && markedOptionsAreSet) {
            lastMessageContainerDiv.querySelectorAll('pre code').forEach((block) => {
                try { hljs.highlightElement(block); } catch (e) { console.error("Error re-highlighting element:", e, block); }
            });
        }

        if (chatHistory && isNearBottom) {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
    }


    // --- LLM Configuration ---
    function updateConditionalOptions() {
        if (!llmApiSelect) return;
        const selectedApi = llmApiSelect.value;
        document.querySelectorAll('#llm-config-form .conditional-options').forEach(div => {
            div.style.display = 'none';
        });
        if (selectedApi) {
            const targetDivId = `${selectedApi}-options`; // Assumes llmApiSelect.value matches part of the div ID
            const optionsDiv = document.getElementById(targetDivId);
            if (optionsDiv) {
                optionsDiv.style.display = 'block';
            } else {
                console.warn(`updateConditionalOptions: Could not find options div for ID: ${targetDivId}`);
            }
        }
        // Optionally hide API selection warning if one is shown
        if (selectedApi) hideApiSelectionWarning();
    }

    function showApiSelectionWarning() { if (llmApiSelect) llmApiSelect.classList.add('input-error'); }
    function hideApiSelectionWarning() { if (llmApiSelect) llmApiSelect.classList.remove('input-error'); }

    function getLlmConfiguration() {
        if (!llmApiSelect) {
            console.error("getLlmConfiguration: llmApiSelect element not found!");
            return { api_type: null }; // Return a minimal config to prevent further errors
        }
        const selectedApi = llmApiSelect.value;
        const config = {
            api_type: selectedApi,
            temperature: parseFloat(temperatureInput.value) || 0.2, // Ensure temperatureInput is defined
            max_tokens: parseInt(maxTokensInput.value) || 4096    // Ensure maxTokensInput is defined
        };

        if (selectedApi) { // Only populate specific options if an API is selected
            const optionsContainer = document.getElementById(`${selectedApi}-options`);
            if (optionsContainer) {
                optionsContainer.querySelectorAll('input, select, textarea').forEach(el => {
                    if (el.name && el.id) { // Use el.name or el.id for the key in config
                        let key = el.name || el.id; // Prefer name if available
                        // Prefix with selectedApi to avoid clashes if IDs are not unique across sections (though they should be)
                        // For simplicity, assuming IDs like 'openai-api-key' are unique enough or mapped by backend
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
                 // Specific handling for known fields if IDs/names are not directly mapping to backend keys
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
                    config.openai_reasoning_model = document.getElementById('openai-reasoning-model')?.checked;
                } else if (selectedApi === "azure_openai") {
                    config.azure_openai_api_key = document.getElementById('azure-openai-api-key')?.value;
                    config.azure_endpoint = document.getElementById('azure-endpoint')?.value;
                    config.azure_api_version = document.getElementById('azure-api-version')?.value;
                    config.azure_deployment_name = document.getElementById('azure-deployment-name')?.value;
                    config.azure_reasoning_model = document.getElementById('azure-reasoning-model')?.checked;
                } else if (selectedApi === "litellm") {
                    config.litellm_model = document.getElementById('litellm-model')?.value;
                    config.litellm_api_key = document.getElementById('litellm-api-key')?.value;
                    config.litellm_base_url = document.getElementById('litellm-base-url')?.value;
                }
            }
        }
        return config;
    }

    // --- Chat Streaming & Backend Communication ---
    function startChatStream(currentMessages, llmConfig) {
        userInput.disabled = true;
        sendButton.disabled = true;
        
        // This variable will hold the progressively built text content of the assistant's response
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
            let sseBuffer = ''; // Buffer for potentially incomplete SSE messages
    
            function processStream({ done, value }) {
                if (done) {
                    if (assistantMessageContent) { // If we received any content
                        const lastMsgIndex = conversationHistory.length - 1;
                        if (lastMsgIndex >= 0 && conversationHistory[lastMsgIndex].role === 'assistant') {
                            // Update the placeholder in history with the final content
                            conversationHistory[lastMsgIndex].content = assistantMessageContent;
                        } else { 
                            // This case should ideally not be hit if placeholder was added correctly
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
                
                // Keep the last (potentially incomplete) line for the next chunk
                sseBuffer = lines.pop(); 
    
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try {
                            const jsonData = JSON.parse(line.substring(6));
                            if (jsonData.text) {
                                if (!firstChunkProcessed) {
                                    assistantMessageContent = jsonData.text; // Start with the first piece of text
                                    firstChunkProcessed = true;
                                } else {
                                    assistantMessageContent += jsonData.text; // Append subsequent pieces
                                }
                                updateLastAssistantMessageDOM(assistantMessageContent); // Update display
                            } else if (jsonData.error) {
                                console.error("Stream error from backend:", jsonData.error);
                                // Display the error in the assistant's message area
                                assistantMessageContent = `**Stream Error:** ${jsonData.error}`;
                                updateLastAssistantMessageDOM(assistantMessageContent);
                                // Optionally, stop further processing by not calling reader.read() or by throwing an error
                            }
                        } catch (e) {
                            // console.warn("Error parsing SSE JSON data:", e, "Original line:", line);
                            // It's possible a non-JSON 'data:' line or other event is received.
                            // If these are expected, they need specific handling.
                        }
                    } else if (line.startsWith('event: end')) { // Example of handling a custom event
                        console.log("Stream indicated end via custom event.");
                        // The 'done' flag from the reader will also handle stream termination.
                    }
                });
    
                return reader.read().then(processStream);
            }
            return reader.read().then(processStream);
        })
        .catch(error => {
            console.error('Chat stream or fetch error:', error);
            // Update the placeholder with the error message
            updateLastAssistantMessageDOM(`**Error:** ${error.message}`);
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
            // Ensure the placeholder in history is also updated or handled
            const lastMsgIndex = conversationHistory.length - 1;
            if (lastMsgIndex >= 0 && conversationHistory[lastMsgIndex].role === 'assistant' && 
                conversationHistory[lastMsgIndex].content === '...') {
                conversationHistory[lastMsgIndex].content = `**Error:** ${error.message}`;
                savePromptEditorState();
            }
        });
    }

    // --- Event Listeners Setup ---
    loadPromptEditorState(); // Load state first

    // LLM API select change
    if (llmApiSelect) {
        llmApiSelect.addEventListener('change', () => {
            updateConditionalOptions();
            savePromptEditorState(); // Save when API selection changes
        });
    }

    // General listener for control panel inputs to save state
    const controlPanelElements = document.querySelectorAll(
        '#llm-config-form select, #llm-config-form input, #llm-config-form textarea'
    );
    controlPanelElements.forEach(element => {
        // Use 'input' for textareas and text-like inputs for immediate saving, 'change' for others
        const eventType = (element.tagName.toLowerCase() === 'textarea' || 
                           (element.type && element.type.match(/text|url|password|number|search|email|tel/))) 
                          ? 'input' : 'change';
        element.addEventListener(eventType, savePromptEditorState);
    });

    // Send button click
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

            // Add user message to DOM and history
            addMessageToChatDOM('user', userText);
            conversationHistory.push({ role: 'user', content: userText });
            userInput.value = ''; // Clear input field

            // Add placeholder for assistant response
            addMessageToChatDOM('assistant', '...');
            // The actual content for this will be updated by the stream.
            // Add placeholder to history; its content will be updated upon stream completion.
            conversationHistory.push({ role: 'assistant', content: '...' }); 
            
            savePromptEditorState(); // Save state with user message and placeholder

            // Pass conversation history *without* the local '...' assistant placeholder for the backend
            startChatStream(conversationHistory.slice(0, -1), llmConfig);
        });
    }

    // User input Enter key press
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent newline in textarea
                sendButton.click(); // Trigger send button click
            }
        });
    }

    // Clear chat button
    if (clearChatButton) {
        clearChatButton.addEventListener('click', () => {
            if (chatHistory) chatHistory.innerHTML = '';
            conversationHistory = [];
            savePromptEditorState(); // Save empty chat
            userInput.disabled = false; // Re-enable inputs
            sendButton.disabled = false;
            hideApiSelectionWarning(); // Clear any API selection warnings
        });
    }

    // Copy button event listener (delegated from chatHistory)
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

                    // Check if the Clipboard API is available and we're in a secure context
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
                            // Alert for modern API failure
                            alert('Failed to copy text. Error: ' + err.message + '\nMake sure you are on a secure connection (HTTPS) or localhost, or check browser permissions.');
                            setTimeout(() => {
                                copyButton.innerHTML = originalIconHTML;
                                copyButton.classList.remove('copied-failed');
                                copyButton.disabled = false;
                            }, 3000);
                        });
                    } else {
                        // Fallback for when navigator.clipboard.writeText is not available
                        console.warn('navigator.clipboard.writeText API not available. Trying legacy copy command.');
                        try {
                            const textArea = document.createElement("textarea");
                            textArea.value = rawTextToCopy;
                            // Prevent visual disruption
                            textArea.style.position = "fixed"; // Use fixed to ensure it's out of viewport flow
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

    // --- Final Initialization Step ---
    window.promptEditorInitialized = true;
    console.log("Prompt Editor UI and event listeners initialized.");

} // End of initializePromptEditor

// The call to initializePromptEditor should be managed by app_shell.html
// when DOM and dependencies (marked, hljs) are ready.