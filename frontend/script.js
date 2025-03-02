document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImage = document.getElementById('removeImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');
    const loader = document.getElementById('loader');
    
    let selectedFile = null;
    
    // Handle click on upload area
    uploadArea.addEventListener('click', function() {
        if (!previewContainer.hidden) return;
        imageInput.click();
    });
    
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#3498db';
        uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'transparent';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'transparent';
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Handle file selection
    imageInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFile(this.files[0]);
        }
    });
    
    // Handle remove image button
    removeImage.addEventListener('click', function(e) {
        e.stopPropagation();
        resetUpload();
    });
    
    // Handle analyze button
    analyzeBtn.addEventListener('click', function() {
        if (!selectedFile) return;
        
        // Show loader
        loader.hidden = false;
        resultContainer.hidden = true;
        analyzeBtn.disabled = true;
        
        // Create form data
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Send request to backend
        fetch('http://localhost:8000/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Display result
            displayResult(data);
        })
        .catch(error => {
            console.error('Error:', error);
            displayResult({ error: 'Failed to analyze image. Please try again.' });
        })
        .finally(() => {
            // Hide loader
            loader.hidden = true;
            analyzeBtn.disabled = false;
        });
    });
    
    // Helper functions
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            uploadPlaceholder.hidden = true;
            previewContainer.hidden = false;
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
    
    function resetUpload() {
        selectedFile = null;
        imageInput.value = '';
        uploadPlaceholder.hidden = false;
        previewContainer.hidden = true;
        analyzeBtn.disabled = true;
        resultContainer.hidden = true;
    }
    
    function displayResult(result) {
        resultContainer.hidden = false;
        
        if (result.error) {
            resultContent.innerHTML = `<p class="error">${result.error}</p>`;
            return;
        }
        
        // Format the result based on your model's output
        let html = '';
        
        if (result.class !== undefined) {
            html += `<p><strong>Class:</strong> ${result.class}</p>`;
        }
        
        if (result.confidence !== undefined) {
            const confidence = (result.confidence * 100).toFixed(2);
            html += `<p><strong>Confidence:</strong> ${confidence}%</p>`;
        }
        
        if (result.description) {
            html += `<p><strong>Description:</strong> ${result.description}</p>`;
        }
        
        resultContent.innerHTML = html || JSON.stringify(result, null, 2);
    }
});
