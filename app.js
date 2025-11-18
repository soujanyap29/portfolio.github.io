// Protein Localization Frontend Application

// Global state
let uploadedFile = null;
let processingResults = null;

// Class names for protein localization
const CLASS_NAMES = [
    'Nucleus',
    'Mitochondria',
    'Endoplasmic Reticulum',
    'Golgi Apparatus',
    'Cytoplasm'
];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupFileUpload();
    setupDragAndDrop();
});

// Setup file upload functionality
function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file && (file.name.endsWith('.tif') || file.name.endsWith('.tiff'))) {
            handleFileUpload(file);
        } else {
            alert('Please upload a TIFF file (.tif or .tiff)');
        }
    });
}

// Handle file upload
function handleFileUpload(file) {
    uploadedFile = file;
    
    // Show file info
    const fileInfo = document.getElementById('fileInfo');
    fileInfo.innerHTML = `
        <strong>File:</strong> ${file.name}<br>
        <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
        <strong>Type:</strong> ${file.type || 'TIFF Image'}
    `;
    fileInfo.classList.add('show');
    
    // Start processing
    setTimeout(() => {
        processImage();
    }, 1000);
}

// Simulate image processing
function processImage() {
    // Hide upload section and show processing
    document.querySelector('.upload-section').style.display = 'none';
    document.getElementById('processingSection').style.display = 'block';
    
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    const steps = [
        { progress: 20, text: 'Loading TIFF image...' },
        { progress: 40, text: 'Segmenting cellular structures...' },
        { progress: 60, text: 'Constructing graph representation...' },
        { progress: 80, text: 'Running Graph CNN model...' },
        { progress: 100, text: 'Generating visualizations...' }
    ];
    
    let currentStep = 0;
    
    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            progressFill.style.width = steps[currentStep].progress + '%';
            progressText.textContent = steps[currentStep].text;
            currentStep++;
        } else {
            clearInterval(interval);
            setTimeout(() => {
                showResults();
            }, 500);
        }
    }, 1500);
}

// Show results
function showResults() {
    // Hide processing section
    document.getElementById('processingSection').style.display = 'none';
    
    // Generate synthetic results
    const results = generateSyntheticResults();
    processingResults = results;
    
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Update prediction
    document.getElementById('predictionLabel').textContent = results.prediction;
    document.getElementById('confidenceValue').textContent = 
        (results.confidence * 100).toFixed(1) + '%';
    
    // Update detailed results
    document.getElementById('numRegions').textContent = results.numRegions;
    document.getElementById('avgArea').textContent = results.avgArea.toFixed(2) + ' px²';
    document.getElementById('meanIntensity').textContent = results.meanIntensity.toFixed(3);
    document.getElementById('graphNodes').textContent = results.graphNodes;
    document.getElementById('graphEdges').textContent = results.graphEdges;
    document.getElementById('processingTime').textContent = results.processingTime + ' s';
    
    // Draw visualizations
    drawOriginalImage();
    drawSegmentedImage();
    drawGraph(results.graphNodes, results.graphEdges);
    drawFeatureChart();
    drawProbabilityBars(results.probabilities);
}

// Generate synthetic results for demonstration
function generateSyntheticResults() {
    const predictionIdx = Math.floor(Math.random() * CLASS_NAMES.length);
    
    // Generate probabilities with highest for predicted class
    const probabilities = CLASS_NAMES.map((name, idx) => {
        if (idx === predictionIdx) {
            return 0.7 + Math.random() * 0.25; // 70-95% for predicted class
        }
        return Math.random() * 0.2; // 0-20% for others
    });
    
    // Normalize probabilities
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalizedProbs = probabilities.map(p => p / sum);
    
    return {
        prediction: CLASS_NAMES[predictionIdx],
        confidence: normalizedProbs[predictionIdx],
        probabilities: normalizedProbs,
        numRegions: Math.floor(Math.random() * 10) + 5,
        avgArea: Math.random() * 500 + 200,
        meanIntensity: Math.random() * 0.5 + 0.4,
        graphNodes: Math.floor(Math.random() * 10) + 5,
        graphEdges: Math.floor(Math.random() * 15) + 8,
        processingTime: (Math.random() * 3 + 2).toFixed(2)
    };
}

// Draw original image (synthetic)
function drawOriginalImage() {
    const canvas = document.getElementById('originalCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 300;
    
    // Create gradient background
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, '#1e3c72');
    gradient.addColorStop(1, '#2a5298');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw some cellular structures
    for (let i = 0; i < 8; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const radius = Math.random() * 30 + 20;
        
        const cellGradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        cellGradient.addColorStop(0, 'rgba(100, 200, 255, 0.8)');
        cellGradient.addColorStop(1, 'rgba(100, 200, 255, 0.2)');
        
        ctx.fillStyle = cellGradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Draw segmented image (synthetic)
function drawSegmentedImage() {
    const canvas = document.getElementById('segmentedCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 300;
    
    // Background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw segmented regions with different colors
    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'];
    
    for (let i = 0; i < 8; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const radius = Math.random() * 30 + 20;
        
        ctx.fillStyle = colors[i % colors.length];
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Add label
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText((i + 1).toString(), x, y + 5);
    }
}

// Draw graph visualization
function drawGraph(numNodes, numEdges) {
    const svg = document.getElementById('graphSvg');
    svg.innerHTML = ''; // Clear existing
    
    const width = 400;
    const height = 300;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = 100;
    
    // Generate node positions in a circle
    const nodes = [];
    for (let i = 0; i < numNodes; i++) {
        const angle = (i / numNodes) * 2 * Math.PI;
        nodes.push({
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle),
            id: i + 1
        });
    }
    
    // Draw edges
    const colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'];
    for (let i = 0; i < numEdges; i++) {
        const fromIdx = Math.floor(Math.random() * numNodes);
        const toIdx = Math.floor(Math.random() * numNodes);
        
        if (fromIdx !== toIdx) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', nodes[fromIdx].x);
            line.setAttribute('y1', nodes[fromIdx].y);
            line.setAttribute('x2', nodes[toIdx].x);
            line.setAttribute('y2', nodes[toIdx].y);
            line.setAttribute('stroke', colors[i % colors.length]);
            line.setAttribute('stroke-width', '2');
            line.setAttribute('opacity', '0.6');
            svg.appendChild(line);
        }
    }
    
    // Draw nodes
    nodes.forEach((node, idx) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', node.x);
        circle.setAttribute('cy', node.y);
        circle.setAttribute('r', '15');
        circle.setAttribute('fill', colors[idx % colors.length]);
        circle.setAttribute('stroke', '#fff');
        circle.setAttribute('stroke-width', '2');
        svg.appendChild(circle);
        
        // Add label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', node.x);
        text.setAttribute('y', node.y + 5);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#fff');
        text.setAttribute('font-size', '12');
        text.setAttribute('font-weight', 'bold');
        text.textContent = node.id;
        svg.appendChild(text);
    });
}

// Draw feature chart
function drawFeatureChart() {
    const canvas = document.getElementById('featureCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 300;
    
    const features = ['Area', 'Intensity', 'Eccentricity', 'Solidity'];
    const values = [0.75, 0.85, 0.62, 0.88];
    const colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'];
    
    const barWidth = 60;
    const spacing = (canvas.width - barWidth * features.length) / (features.length + 1);
    const maxHeight = canvas.height - 60;
    
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    
    features.forEach((feature, idx) => {
        const x = spacing + idx * (barWidth + spacing);
        const height = values[idx] * maxHeight;
        const y = canvas.height - height - 30;
        
        // Draw bar
        ctx.fillStyle = colors[idx];
        ctx.fillRect(x, y, barWidth, height);
        
        // Draw label
        ctx.fillStyle = '#333';
        ctx.fillText(feature, x + barWidth / 2, canvas.height - 10);
        
        // Draw value
        ctx.fillStyle = '#fff';
        ctx.fillText(values[idx].toFixed(2), x + barWidth / 2, y - 10);
    });
}

// Draw probability bars
function drawProbabilityBars(probabilities) {
    const container = document.getElementById('probabilitiesBars');
    container.innerHTML = '';
    
    CLASS_NAMES.forEach((name, idx) => {
        const prob = probabilities[idx];
        const percentage = (prob * 100).toFixed(1);
        
        const barHTML = `
            <div class="probability-bar">
                <div class="probability-label">
                    <span>${name}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="probability-fill-container">
                    <div class="probability-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
        
        container.innerHTML += barHTML;
    });
}

// Download results
function downloadResults() {
    if (!processingResults) return;
    
    const results = {
        file: uploadedFile ? uploadedFile.name : 'unknown',
        timestamp: new Date().toISOString(),
        prediction: processingResults.prediction,
        confidence: processingResults.confidence,
        probabilities: CLASS_NAMES.map((name, idx) => ({
            class: name,
            probability: processingResults.probabilities[idx]
        })),
        details: {
            numRegions: processingResults.numRegions,
            avgArea: processingResults.avgArea,
            meanIntensity: processingResults.meanIntensity,
            graphNodes: processingResults.graphNodes,
            graphEdges: processingResults.graphEdges,
            processingTime: processingResults.processingTime
        }
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], 
                          { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'protein_localization_results.json';
    a.click();
    URL.revokeObjectURL(url);
}

// Reset application
function resetApp() {
    uploadedFile = null;
    processingResults = null;
    
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('processingSection').style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    
    document.getElementById('fileInfo').classList.remove('show');
    document.getElementById('fileInput').value = '';
    
    // Reset progress
    document.getElementById('progressFill').style.width = '0%';
}
