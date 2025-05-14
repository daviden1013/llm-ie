document.addEventListener('DOMContentLoaded', () => {
    // --- UI Element Getters ---
    const llmieUploadArea = document.getElementById('rv-llmie-upload-area');
    const llmieFileInput = document.getElementById('rv-llmie-file-input');
    const loadedFilenameSpan = document.getElementById('rv-loaded-filename');
    const colorKeyGroupDiv = document.getElementById('rv-color-key-group');
    const colorKeySelect = document.getElementById('rv-color-key-select');
    const renderButton = document.getElementById('render-viz-btn');
    const vizOutputDiv = document.getElementById('visualization-output');

    let uploadedLlmieData = null; // To store data from the uploaded .llmie file

    // --- Helper Functions ---
    function resetUploadState(errorMessage = null) {
        loadedFilenameSpan.textContent = '';
        if (llmieFileInput) { // Check if element exists
            llmieFileInput.value = ''; // Clear the file input
        }
        // if (colorKeyGroupDiv) colorKeyGroupDiv.style.display = 'none'; // MODIFIED: Keep the group visible
        if (colorKeySelect) colorKeySelect.innerHTML = '<option value="" disabled selected>-- Upload a file to see options --</option>'; // MODIFIED: Update placeholder
        
        uploadedLlmieData = null;
        
        if (vizOutputDiv) { // Check if element exists
            vizOutputDiv.innerHTML = `<p>${errorMessage ? errorMessage : 'Upload an .llmie file and click Render.'}</p>`;
        }
    }

    function handleFile(file) {
        if (!file) {
            resetUploadState('No file selected.');
            return;
        }

        if (!file.name.endsWith('.llmie')) {
            resetUploadState('Invalid file type. Please upload an .llmie file.');
            alert('Invalid file type. Please upload an .llmie file.');
            return;
        }

        if (loadedFilenameSpan) loadedFilenameSpan.textContent = `File: ${file.name}`;
        if (vizOutputDiv) vizOutputDiv.innerHTML = '<p>Processing file...</p>'; // Show loading state

        const formData = new FormData();
        formData.append('llmie_file', file);

        fetch('/api/results/process_llmie_data', { 
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { 
                    throw new Error(err.error || `File upload failed with status: ${response.status}`);
                }).catch(() => { 
                    throw new Error(`File upload failed with status: ${response.status} and no error details.`);
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.text !== undefined && data.frames && data.attribute_keys) {
                uploadedLlmieData = {
                    text: data.text,
                    frames: data.frames,
                    relations: data.relations || [], 
                    attribute_keys: data.attribute_keys
                };

                if (colorKeySelect) {
                    if (uploadedLlmieData.attribute_keys.length > 0) {
                        colorKeySelect.innerHTML = '<option value="" selected>None (Default Colors)</option>'; 
                        uploadedLlmieData.attribute_keys.forEach(key => {
                            const option = document.createElement('option');
                            option.value = key;
                            option.textContent = key;
                            colorKeySelect.appendChild(option);
                        });
                    } else {
                        // MODIFIED: Set placeholder if no keys are available
                        colorKeySelect.innerHTML = '<option value="" disabled selected>-- No attributes to color by --</option>';
                    }
                    // MODIFIED: colorKeyGroupDiv should already be visible by HTML default
                    // if (colorKeyGroupDiv) colorKeyGroupDiv.style.display = 'block'; 
                }
                if (vizOutputDiv) vizOutputDiv.innerHTML = '<p>File processed. Click "Render Visualization".</p>';
            } else {
                throw new Error(data.error || 'Invalid data structure received from server after file processing.');
            }
        })
        .catch(error => {
            console.error('Error uploading or processing .llmie file:', error);
            resetUploadState(`Error: ${error.message}`);
        });
    }

    // --- Event Listeners for File Upload ---
    if (llmieUploadArea && llmieFileInput) {
        llmieUploadArea.addEventListener('click', () => {
            llmieFileInput.click(); 
        });

        llmieFileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            handleFile(file);
        });

        llmieUploadArea.addEventListener('dragover', (event) => {
            event.preventDefault(); 
            llmieUploadArea.classList.add('dragover');
        });

        llmieUploadArea.addEventListener('dragleave', () => {
            llmieUploadArea.classList.remove('dragover');
        });

        llmieUploadArea.addEventListener('drop', (event) => {
            event.preventDefault();
            llmieUploadArea.classList.remove('dragover');
            const file = event.dataTransfer.files[0];
            handleFile(file);
        });
    } else {
        console.error("Result Viewer: Critical upload UI elements (upload area or file input) not found.");
    }

    // --- Event Listener for Render Button ---
    if (renderButton) {
        renderButton.addEventListener('click', async () => {
            if (!uploadedLlmieData) {
                alert('Please upload and process an .llmie file first.');
                return;
            }

            const { text, frames, relations } = uploadedLlmieData;
            const colorKey = colorKeySelect ? colorKeySelect.value : null; 

            const payload = {
                text: text,
                frames: frames, 
                relations: relations, 
                vizOptions: {
                    color_attr_key: colorKey || null 
                }
            };

            if (vizOutputDiv) vizOutputDiv.innerHTML = '<p>Rendering...</p>'; 

            try {
                const response = await fetch('/api/results/render', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }

                const result = await response.json();

                if (result.html) {
                    const iframe = document.createElement('iframe');
                    iframe.srcdoc = result.html;
                    iframe.style.width = '100%';
                    iframe.style.height = '100%';
                    iframe.style.border = 'none';
                    if (vizOutputDiv) {
                        vizOutputDiv.innerHTML = '';
                        vizOutputDiv.appendChild(iframe);
                    }
                } else {
                    if (vizOutputDiv) vizOutputDiv.innerHTML = '<p style="color: red;">Error: Received no HTML content from the server.</p>';
                }
            } catch (error) {
                console.error('Error rendering visualization:', error);
                if (vizOutputDiv) vizOutputDiv.innerHTML = `<p style="color: red;">Error rendering visualization: ${error.message}</p>`;
            }
        });
    } else {
        console.error("Result Viewer: Render button not found.");
    }

    // Initial reset when the page loads
    resetUploadState(); 
    // Ensure rv-color-key-group is visible if it was hidden by CSS and not inline style
    if (colorKeyGroupDiv && getComputedStyle(colorKeyGroupDiv).display === 'none') {
        colorKeyGroupDiv.style.display = 'block'; // Or 'flex' or appropriate display type
    }
});