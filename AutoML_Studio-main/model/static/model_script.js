document.addEventListener('DOMContentLoaded', function() {
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadBtnText = document.getElementById('uploadBtnText');
    const uploadSpinner = document.getElementById('uploadSpinner');
    const trainBtn = document.getElementById('trainBtn');
    const trainBtnText = document.getElementById('trainBtnText');
    const trainSpinner = document.getElementById('trainSpinner');
    const exportBtn = document.getElementById('exportBtn');
    const deployBtn = document.getElementById('deployBtn');
    const fullExportBtn = document.getElementById('fullExportBtn');
    const testPredictBtn = document.getElementById('testPredictBtn');
    const generateUIBtn = document.getElementById('generateUIBtn');
    const exportUIBtn = document.getElementById('exportUIBtn');
    const copyCodeBtn = document.getElementById('copyCodeBtn');
    const selectAllFeatures = document.getElementById('selectAllFeatures');
    const deselectAllFeatures = document.getElementById('deselectAllFeatures');
    const uiTheme = document.getElementById('uiTheme');
    const uiLayout = document.getElementById('uiLayout');
    const gridSearchToggle = document.getElementById('gridSearchToggle');
    const preprocessToggle = document.getElementById('preprocessToggle');
    
    let currentFeatures = [];
    let currentTarget = '';
    let currentProblemType = '';
    let currentModelName = '';
    let currentFeatureNames = [];
    let currentColumnTypes = {};
    
    // Event listeners
    uploadBtn.addEventListener('click', uploadDataset);
    trainBtn.addEventListener('click', trainModel);
    exportBtn.addEventListener('click', exportModel);
    deployBtn.addEventListener('click', deployModel);
    fullExportBtn.addEventListener('click', exportFullApp);
    testPredictBtn.addEventListener('click', testPrediction);
    generateUIBtn.addEventListener('click', generateUICode);
    exportUIBtn.addEventListener('click', exportUI);
    copyCodeBtn.addEventListener('click', copyAllToClipboard);
    selectAllFeatures.addEventListener('click', selectAllFeaturesHandler);
    deselectAllFeatures.addEventListener('click', deselectAllFeaturesHandler);
    uiTheme.addEventListener('change', updateUITheme);
    uiLayout.addEventListener('change', updateUILayout);
    
    // Dark theme initialization
    function initializeDarkTheme() {
        document.body.classList.add('dark-theme');
        updateUITheme();
    }
    
    // Dataset upload function
    function uploadDataset() {
        const fileInput = document.getElementById('datasetFile');
        const file = fileInput.files[0];
        
        if (!file) {
            showToast('Please select a file first', 'warning');
            return;
        }
        
        uploadBtn.disabled = true;
        uploadBtnText.textContent = 'Uploading...';
        uploadSpinner.classList.remove('d-none');
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            uploadBtn.disabled = false;
            uploadBtnText.textContent = 'Upload Dataset';
            uploadSpinner.classList.add('d-none');
            
            if (data.error) {
                showToast(data.error, 'danger');
                return;
            }
            
            // Display data preview with dark table styling
            document.getElementById('dataPreview').innerHTML = `
                <div class="table-container">
                    <table class="table table-dark table-striped table-hover">
                        ${data.preview}
                    </table>
                </div>`;
            
            currentColumnTypes = data.column_types || {};
            
            // Populate target column dropdown
            const targetColumnSelect = document.getElementById('targetColumn');
            targetColumnSelect.innerHTML = data.columns.map(col => 
                `<option value="${col}">${col}</option>`
            ).join('');
            
            // Populate feature columns with dark theme checkboxes
            const featureColumnsDiv = document.getElementById('featureColumns');
            featureColumnsDiv.innerHTML = data.columns.map(col => {
                const isCategorical = currentColumnTypes[col] === 'categorical';
                return `
                    <div class="col-md-4 mb-2 feature-item">
                        <input type="checkbox" class="form-check-input feature-checkbox" 
                               id="feature-${col}" value="${col}" checked>
                        <label class="form-check-label" for="feature-${col}">
                            ${col}
                            <span class="badge ms-2 ${isCategorical ? 'bg-warning text-dark' : 'bg-info'}">
                                ${isCategorical ? 'CAT' : 'NUM'}
                            </span>
                        </label>
                    </div>`;
            }).join('');
            
            currentProblemType = data.problem_type;
            document.getElementById('problemType').innerHTML = `
                <div class="alert alert-info">
                    <i class="bi bi-info-circle"></i> Detected problem type: <strong>${data.problem_type}</strong>
                </div>`;
            
            // Populate model type dropdown based on problem type
            const modelTypeSelect = document.getElementById('modelType');
            const models = currentProblemType === 'classification' ? 
                ['Random Forest', 'Logistic Regression', 'Gradient Boosting', 'SVM', 'K-Nearest Neighbors'] : 
                ['Random Forest', 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Gradient Boosting', 'SVM', 'K-Nearest Neighbors'];
            
            modelTypeSelect.innerHTML = models.map(model => 
                `<option value="${model}">${model}</option>`
            ).join('');
            
            // Show next sections
            document.getElementById('modelTrainingSection').style.display = 'block';
            document.getElementById('modelEvaluationSection').style.display = 'none';
            document.getElementById('uiBuilderSection').style.display = 'none';
            document.getElementById('deploymentSection').style.display = 'none';
            
            showToast('Dataset uploaded successfully!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            uploadBtn.disabled = false;
            uploadBtnText.textContent = 'Upload Dataset';
            uploadSpinner.classList.add('d-none');
            showToast('An error occurred while uploading the file', 'danger');
        });
    }
    
    // Feature selection handlers
    function selectAllFeaturesHandler(e) {
        e.preventDefault();
        document.querySelectorAll('.feature-checkbox').forEach(checkbox => {
            checkbox.checked = true;
        });
    }
    
    function deselectAllFeaturesHandler(e) {
        e.preventDefault();
        document.querySelectorAll('.feature-checkbox').forEach(checkbox => {
            checkbox.checked = false;
        });
    }
    
    // Model training function
    function trainModel() {
        const targetColumn = document.getElementById('targetColumn').value;
        const modelType = document.getElementById('modelType').value;
        const useGridSearch = gridSearchToggle.checked;
        const preprocess = preprocessToggle.checked;
        
        const features = Array.from(document.querySelectorAll('.feature-checkbox:checked'))
            .map(cb => cb.value)
            .filter(f => f !== targetColumn);
        
        if (features.length === 0) {
            showToast('Please select at least one feature column', 'warning');
            return;
        }
        
        currentFeatures = features;
        currentTarget = targetColumn;
        currentModelName = modelType;
        
        trainBtn.disabled = true;
        trainBtnText.textContent = 'Training...';
        trainSpinner.classList.remove('d-none');
        
        const trainingResult = document.getElementById('trainingResult');
        trainingResult.innerHTML = `
            <div class="progress bg-dark">
                <div class="progress-bar progress-bar-striped progress-bar-animated bg-primary" 
                     role="progressbar" style="width: 100%"></div>
            </div>
            <div class="text-center mt-2 text-white-50">
                Training model with ${features.length} features...
            </div>`;
        
        fetch('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target: targetColumn,
                features: features,
                model: modelType,
                problem_type: currentProblemType,
                use_grid_search: useGridSearch,
                preprocess: preprocess
            })
        })
        .then(response => response.json())
        .then(data => {
            trainBtn.disabled = false;
            trainBtnText.textContent = 'Train Model';
            trainSpinner.classList.add('d-none');
            
            if (data.error) {
                trainingResult.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="bi bi-exclamation-triangle"></i> ${data.error}
                    </div>`;
                showToast('Training failed', 'danger');
                return;
            }
            
            currentFeatureNames = data.feature_names || features;
            
            trainingResult.innerHTML = `
                <div class="alert alert-success">
                    <i class="bi bi-check-circle"></i> Model trained successfully!
                </div>`;
            
            // Update evaluation metrics
            document.getElementById('primaryMetricTitle').textContent = data.metric_name;
            document.getElementById('primaryMetricValue').textContent = data.score.toFixed(4);
            document.getElementById('secondaryMetricTitle').textContent = data.secondary_metric;
            document.getElementById('secondaryMetricValue').textContent = data.secondary_score.toFixed(4);
            
            // Display best parameters if available
            if (Object.keys(data.best_params).length > 0) {
                let paramsHtml = '<h5 class="mt-3">Optimal Parameters:</h5><ul class="list-group">';
                for (const key in data.best_params) {
                    paramsHtml += `
                        <li class="list-group-item bg-dark text-white">
                            <strong>${key.replace('model__', '')}:</strong> ${data.best_params[key]}
                        </li>`;
                }
                paramsHtml += '</ul>';
                document.getElementById('bestParamsContainer').innerHTML = paramsHtml;
            }
            
            // Build prediction form with dark theme
            buildPredictionForm(currentFeatureNames);
            
            // Show evaluation and deployment sections
            document.getElementById('modelEvaluationSection').style.display = 'block';
            document.getElementById('uiBuilderSection').style.display = 'block';
            document.getElementById('deploymentSection').style.display = 'block';
            
            showToast('Model trained successfully!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            trainBtn.disabled = false;
            trainBtnText.textContent = 'Train Model';
            trainSpinner.classList.add('d-none');
            trainingResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i> An error occurred during training
                </div>`;
            showToast('Training failed', 'danger');
        });
    }
    
    // Build prediction form with dark theme support
    function buildPredictionForm(features) {
        const form = document.getElementById('predictionForm');
        form.innerHTML = '';
        
        features.forEach(feature => {
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group mb-3';
            
            const label = document.createElement('label');
            label.textContent = feature;
            label.htmlFor = `input-${feature}`;
            label.className = 'form-label text-white';
            
            let input;
            if (currentColumnTypes[feature] === 'categorical') {
                input = document.createElement('select');
                input.className = 'form-select form-control-dark';
                input.id = `input-${feature}`;
                
                ['Category 1', 'Category 2'].forEach((cat, i) => {
                    const option = document.createElement('option');
                    option.value = `category${i+1}`;
                    option.textContent = cat;
                    input.appendChild(option);
                });
            } else {
                input = document.createElement('input');
                input.type = 'number';
                input.className = 'form-control form-control-dark';
                input.id = `input-${feature}`;
                input.placeholder = `Enter ${feature}`;
                input.step = 'any';
            }
            
            input.required = true;
            
            formGroup.appendChild(label);
            formGroup.appendChild(input);
            form.appendChild(formGroup);
        });
        
        updateUITheme();
        updateUILayout();
    }
    
    // Test prediction function
    function testPrediction() {
        const form = document.getElementById('predictionForm');
        const inputs = form.querySelectorAll('input, select');
        const features = {};
        
        let isValid = true;
        inputs.forEach(input => {
            if (!input.value) {
                isValid = false;
                input.classList.add('is-invalid');
            } else {
                input.classList.remove('is-invalid');
                features[input.id.replace('input-', '')] = 
                    input.type === 'number' ? parseFloat(input.value) : input.value;
            }
        });
        
        if (!isValid) {
            showToast('Please fill in all fields', 'warning');
            return;
        }
        
        const featureArray = currentFeatureNames.map(f => features[f]);
        
        document.getElementById('predictionResult').innerHTML = `
            <div class="alert alert-info">
                <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                Making prediction...
            </div>`;
        
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: featureArray })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('predictionResult').innerHTML = `
                    <div class="alert alert-danger">
                        <i class="bi bi-exclamation-triangle"></i> ${data.error}
                    </div>`;
                showToast('Prediction failed', 'danger');
                return;
            }
            
            document.getElementById('predictionResult').innerHTML = `
                <div class="alert alert-success">
                    <i class="bi bi-check-circle"></i> Prediction result: <strong>${data.prediction}</strong>
                </div>`;
            showToast('Prediction successful!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('predictionResult').innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i> An error occurred during prediction
                </div>`;
            showToast('Prediction failed', 'danger');
        });
    }
    
    // UI theme and layout updates
    function updateUITheme() {
        const uiPreview = document.getElementById('uiPreview');
        uiPreview.className = 'mb-4 p-3 border rounded';
        
        // Remove all theme classes first
        uiPreview.classList.remove('dark-theme', 'light-theme', 'modern-theme');
        
        // Add selected theme class
        uiPreview.classList.add(`${uiTheme.value}-theme`);
        
        // Update form controls
        const formControls = uiPreview.querySelectorAll('.form-control, .form-select');
        formControls.forEach(control => {
            control.className = control.className.replace(/\bform-control-\w+\b/g, '');
            if (uiTheme.value === 'dark') {
                control.classList.add('form-control-dark');
            } else {
                control.classList.add('form-control');
            }
        });
    }
    
    function updateUILayout() {
        const form = document.getElementById('predictionForm');
        form.className = '';
        
        switch (uiLayout.value) {
            case 'horizontal':
                form.classList.add('horizontal-form');
                break;
            case 'grid':
                form.classList.add('grid-form');
                break;
            default:
                form.classList.add('vertical-form');
                break;
        }
    }
    
    // Model export functions
    function exportModel() {
        exportBtn.disabled = true;
        exportBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span> Exporting...';
        
        fetch('/export')
        .then(response => {
            if (!response.ok) throw new Error('Export failed');
            return response.blob();
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ai_model_package.zip';
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(url);
            a.remove();
            
            exportBtn.disabled = false;
            exportBtn.innerHTML = 'Export Model';
            showToast('Model exported successfully!', 'success');
        })
        .catch(error => {
            console.error('Export error:', error);
            exportBtn.disabled = false;
            exportBtn.innerHTML = 'Export Model';
            showToast('Export failed: ' + error.message, 'danger');
        });
    }
    
    function deployModel() {
        deployBtn.disabled = true;
        deployBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span> Preparing...';
        
        const deploymentResult = document.getElementById('deploymentResult');
        deploymentResult.innerHTML = `
            <div class="progress bg-dark mt-2">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: 100%"></div>
            </div>`;
        
        fetch('/deploy', { method: 'POST' })
        .then(response => {
            if (!response.ok) throw new Error('Deployment failed');
            return response.blob();
        })
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'model_deployment_package.zip';
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(url);
            a.remove();
            
            deployBtn.disabled = false;
            deployBtn.innerHTML = 'Deploy as API';
            deploymentResult.innerHTML = `
                <div class="alert alert-success mt-2">
                    <i class="bi bi-check-circle"></i> Deployment package downloaded successfully!
                </div>`;
            showToast('Deployment package ready!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            deployBtn.disabled = false;
            deployBtn.innerHTML = 'Deploy as API';
            deploymentResult.innerHTML = `
                <div class="alert alert-danger mt-2">
                    <i class="bi bi-exclamation-triangle"></i> An error occurred during deployment
                </div>`;
            showToast('Deployment failed', 'danger');
        });
    }
    
    function exportFullApp() {
        fullExportBtn.disabled = true;
        fullExportBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span> Preparing...';
        
        // First generate the UI code
        generateUICode();
        
        // Get all the generated code
        const htmlCode = document.getElementById('uiCodeHTML').value;
        const jsCode = document.getElementById('uiCodeJS').value;
        const cssCode = document.getElementById('uiCodeCSS').value;
        
        // Create a zip file with all components
        const zip = new JSZip();
        
        // Add model files
        fetch('/export')
        .then(response => {
            if (!response.ok) throw new Error('Failed to export model');
            return response.blob();
        })
        .then(modelBlob => {
            return JSZip.loadAsync(modelBlob).then(modelZip => {
                modelZip.forEach((relativePath, file) => {
                    zip.file(relativePath, file.async('uint8array'));
                });
                
                // Add UI files
                zip.file("templates/index.html", htmlCode);
                zip.file("static/script.js", jsCode);
                zip.file("static/style.css", cssCode);
                
                // Add Flask app file
                zip.file("app.py", 
`from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load the trained model
model = joblib.load(Path('model') / 'model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data['features']
        input_data = pd.DataFrame([features], columns=model.feature_names_in_)
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction[0].tolist() if hasattr(prediction[0], 'tolist') else str(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)`);

                // Add requirements
                zip.file("requirements.txt", "flask\nscikit-learn\npandas\nnumpy");
                
                return zip.generateAsync({type: 'blob'});
            });
        })
        .then(content => {
            const url = URL.createObjectURL(content);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'full_ai_application.zip';
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(url);
            a.remove();
            
            fullExportBtn.disabled = false;
            fullExportBtn.innerHTML = 'Export Full App';
            showToast('Full application exported successfully!', 'success');
        })
        .catch(error => {
            console.error('Full export error:', error);
            fullExportBtn.disabled = false;
            fullExportBtn.innerHTML = 'Export Full App';
            showToast('Export failed: ' + error.message, 'danger');
        });
    }
    
    // UI code generation
    function generateUICode() {
        try {
            const form = document.getElementById('predictionForm');
            const formHTML = form.innerHTML;
            const theme = uiTheme.value;
            const layout = uiLayout.value;
            
            // Generate HTML code
            const htmlCode = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${currentModelName} Model Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css" rel="stylesheet">
    <link href="static/style.css" rel="stylesheet">
    <style>
        body.${theme}-theme {
            background-color: ${theme === 'dark' ? '#1a1a1a' : theme === 'light' ? '#f8f9fa' : '#1e2a3a'};
            color: ${theme === 'light' ? '#212529' : '#ffffff'};
            padding: 20px;
        }
        .card {
            max-width: 800px;
            margin: 0 auto;
            background-color: ${theme === 'dark' ? '#2d2d2d' : theme === 'light' ? '#ffffff' : '#2a3a4a'};
            color: ${theme === 'light' ? '#212529' : '#ffffff'};
        }
        .form-control, .form-select {
            background-color: ${theme === 'dark' ? '#333333' : '#ffffff'};
            color: ${theme === 'dark' ? '#ffffff' : '#212529'};
            border-color: ${theme === 'dark' ? '#555555' : '#ced4da'};
        }
    </style>
</head>
<body class="${theme}-theme">
    <div class="container mt-5">
        <div class="card shadow">
            <div class="card-header bg-primary text-white">
                <h2 class="mb-0">${currentModelName} Model Prediction</h2>
            </div>
            <div class="card-body">
                <form id="predictionForm" class="${layout}-form">
                    ${formHTML}
                </form>
                <button id="predictBtn" class="btn btn-primary mt-3">
                    <span id="predictBtnText">Predict</span>
                    <span id="predictSpinner" class="spinner-border spinner-border-sm d-none" role="status"></span>
                </button>
                <div id="predictionResult" class="mt-3"></div>
            </div>
        </div>
    </div>

    <script src="static/script.js"></script>
</body>
</html>`;

            // Generate JavaScript code
            const jsCode = `document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predictBtn');
    const predictBtnText = document.getElementById('predictBtnText');
    const predictSpinner = document.getElementById('predictSpinner');
    
    predictBtn.addEventListener('click', function() {
        const form = document.getElementById('predictionForm');
        const inputs = form.querySelectorAll('input, select');
        const features = {};
        
        let isValid = true;
        inputs.forEach(input => {
            if (!input.value) {
                isValid = false;
                input.classList.add('is-invalid');
            } else {
                input.classList.remove('is-invalid');
                features[input.id.replace('input-', '')] = 
                    input.type === 'number' ? parseFloat(input.value) : input.value;
            }
        });
        
        if (!isValid) {
            alert('Please fill in all fields');
            return;
        }
        
        const featureArray = [${currentFeatureNames.map(f => `features['${f}']`).join(', ')}];
        
        predictBtn.disabled = true;
        predictBtnText.textContent = 'Predicting...';
        predictSpinner.classList.remove('d-none');
        
        document.getElementById('predictionResult').innerHTML = 
            '<div class="alert alert-info">' +
            '<div class="spinner-border spinner-border-sm me-2" role="status"></div>' +
            'Making prediction...' +
            '</div>';
        
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: featureArray })
        })
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            predictBtn.disabled = false;
            predictBtnText.textContent = 'Predict';
            predictSpinner.classList.add('d-none');
            
            if (data.error) {
                document.getElementById('predictionResult').innerHTML = 
                    '<div class="alert alert-danger">' +
                    '<i class="bi bi-exclamation-triangle"></i> ' + data.error +
                    '</div>';
                return;
            }
            
            document.getElementById('predictionResult').innerHTML = 
                '<div class="alert alert-success">' +
                '<i class="bi bi-check-circle"></i> Prediction result: <strong>' + data.prediction + '</strong>' +
                '</div>';
        })
        .catch(error => {
            console.error('Error:', error);
            predictBtn.disabled = false;
            predictBtnText.textContent = 'Predict';
            predictSpinner.classList.add('d-none');
            document.getElementById('predictionResult').innerHTML = 
                '<div class="alert alert-danger">' +
                '<i class="bi bi-exclamation-triangle"></i> An error occurred during prediction' +
                '</div>';
        });
    });
});`;

            // Generate CSS code
            const cssCode = `/* Base styles */
body {
    padding: 20px;
}

.card {
    max-width: 800px;
    margin: 0 auto;
}

/* Theme styles */
.dark-theme {
    background-color: #1a1a1a;
    color: #ffffff;
}

.dark-theme .card {
    background-color: #2d2d2d;
    border-color: #444;
}

.dark-theme .form-control, 
.dark-theme .form-select {
    background-color: #333;
    color: #ffffff;
    border-color: #555;
}

.light-theme {
    background-color: #f8f9fa;
    color: #212529;
}

.modern-theme {
    background-color: #1e2a3a;
    color: #ffffff;
}

.modern-theme .card {
    background-color: #2a3a4a;
    border: none;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

/* Form layout styles */
.vertical-form .form-group {
    margin-bottom: 1rem;
}

.horizontal-form .form-group {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
}

.horizontal-form .form-group label {
    flex: 0 0 150px;
    margin-right: 10px;
}

.horizontal-form .form-group input,
.horizontal-form .form-group select {
    flex: 1;
}

.grid-form {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .horizontal-form .form-group {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .horizontal-form .form-group label {
        margin-bottom: 0.5rem;
        margin-right: 0;
    }
}`;

            // Set the generated code in the UI
            document.getElementById('uiCodeHTML').value = htmlCode;
            document.getElementById('uiCodeJS').value = jsCode;
            document.getElementById('uiCodeCSS').value = cssCode;
            document.getElementById('uiCodeContainer').style.display = 'block';
            
            // Activate the first tab
            const firstTab = document.querySelector('#codeTabs .nav-link');
            if (firstTab) {
                new bootstrap.Tab(firstTab).show();
            }
            
            showToast('UI code generated successfully!', 'success');
        } catch (error) {
            console.error('Error generating UI code:', error);
            showToast('Failed to generate UI code: ' + error.message, 'danger');
        }
    }
    
    // Export UI function
    function exportUI() {
        generateUICode();
        
        const htmlCode = document.getElementById('uiCodeHTML').value;
        const jsCode = document.getElementById('uiCodeJS').value;
        const cssCode = document.getElementById('uiCodeCSS').value;
        
        const zip = new JSZip();
        zip.file("index.html", htmlCode);
        zip.file("script.js", jsCode);
        zip.file("style.css", cssCode);
        
        zip.generateAsync({type: 'blob'}).then(content => {
            const url = URL.createObjectURL(content);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'model_ui.zip';
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(url);
            a.remove();
            
            showToast('UI exported successfully!', 'success');
        });
    }
    
    // Copy to clipboard function
    function copyAllToClipboard() {
        const htmlCode = document.getElementById('uiCodeHTML').value;
        const jsCode = document.getElementById('uiCodeJS').value;
        const cssCode = document.getElementById('uiCodeCSS').value;
        
        const fullCode = `<!-- HTML -->
${htmlCode}

<!-- JavaScript -->
<script>
${jsCode}
</script>

<!-- CSS -->
<style>
${cssCode}
</style>`;
        
        navigator.clipboard.writeText(fullCode)
            .then(() => showToast('All code copied to clipboard!', 'success'))
            .catch(err => {
                console.error('Failed to copy: ', err);
                showToast('Failed to copy code', 'danger');
            });
    }
    
    // Toast notification function
    function showToast(message, type) {
        const toastContainer = document.createElement('div');
        const typeClasses = {
            'success': 'bg-success text-white',
            'danger': 'bg-danger text-white',
            'warning': 'bg-warning text-dark',
            'info': 'bg-info text-white'
        };
        
        toastContainer.className = `toast align-items-center ${typeClasses[type] || 'bg-primary text-white'} border-0`;
        toastContainer.setAttribute('role', 'alert');
        toastContainer.setAttribute('aria-live', 'assertive');
        toastContainer.setAttribute('aria-atomic', 'true');
        
        toastContainer.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast" aria-label="Close"></button>
            </div>`;
        
        document.body.appendChild(toastContainer);
        const toast = new bootstrap.Toast(toastContainer);
        toast.show();
        
        toastContainer.addEventListener('hidden.bs.toast', () => {
            document.body.removeChild(toastContainer);
        });
    }
    
    // Initialize dark theme
    initializeDarkTheme();
});