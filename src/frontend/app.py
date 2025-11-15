"""
Flask Web Interface for Protein Localization Pipeline
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import numpy as np
import json
import pickle
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.preprocess import PreprocessingPipeline
from graph.graph_builder import GraphBuilder
from models.train import GraphCNN, ModelTrainer
from visualization.plots import VisualizationSuite

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = '/mnt/d/5TH_SEM/CELLULAR/output/uploads'
app.config['OUTPUT_FOLDER'] = '/mnt/d/5TH_SEM/CELLULAR/output'
app.config['MODEL_PATH'] = '/mnt/d/5TH_SEM/CELLULAR/output/models/graph_cnn.pth'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global model instance
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ALLOWED_EXTENSIONS = {'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model"""
    global model
    try:
        # Initialize model (need to know feature dimensions)
        num_features = 6  # Default from feature extraction
        num_classes = 6  # soma, dendrite, axon, nucleus, synapse, mitochondria
        
        model = GraphCNN(num_features=num_features, num_classes=num_classes)
        trainer = ModelTrainer(model, device=device)
        
        if os.path.exists(app.config['MODEL_PATH']):
            trainer.load_model(app.config['MODEL_PATH'])
            print("Model loaded successfully")
        else:
            print("No trained model found. Using untrained model.")
        
        return trainer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle TIFF file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_file():
    """Process uploaded TIFF file through the full pipeline"""
    try:
        data = request.json
        filepath = data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Initialize components
        preprocessing_pipeline = PreprocessingPipeline(
            input_dir=os.path.dirname(filepath),
            output_dir=app.config['OUTPUT_FOLDER']
        )
        
        graph_builder = GraphBuilder()
        viz_suite = VisualizationSuite(output_dir=app.config['OUTPUT_FOLDER'])
        
        # Step 1: Preprocess and segment
        result = preprocessing_pipeline.process_single(Path(filepath))
        
        if result is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Step 2: Build graph
        G = graph_builder.build_graph_from_features(result)
        pyg_data = graph_builder.to_pytorch_geometric(G)
        
        # Step 3: Predict with model
        trainer = load_model()
        if trainer and trainer.model:
            trainer.model.eval()
            with torch.no_grad():
                pyg_data = pyg_data.to(device)
                output = trainer.model(pyg_data.x, pyg_data.edge_index)
                prediction = output.argmax(dim=1).item()
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        else:
            prediction = 0
            probabilities = np.ones(6) / 6
        
        class_names = ['Soma', 'Dendrite', 'Axon', 'Nucleus', 'Synapse', 'Mitochondria']
        predicted_class = class_names[prediction]
        
        # Step 4: Generate visualizations
        viz_data = {
            'image': result.get('masks'),  # Use masks for now
            'masks': result['masks'],
            'graph': G
        }
        
        # Generate key visualizations
        import tifffile
        original_img = tifffile.imread(filepath)
        if original_img.ndim == 4:
            original_img = np.max(original_img, axis=(0, 1))
        elif original_img.ndim == 3 and original_img.shape[-1] > 3:
            original_img = np.max(original_img, axis=0)
        
        viz_suite.plot_image_overlay(original_img, result['masks'],
                                     title=f"Prediction: {predicted_class}",
                                     save_name=f"{Path(filepath).stem}_overlay.png")
        
        viz_suite.plot_graph_visualization(G,
                                          title=f"Biological Graph - {predicted_class}",
                                          save_name=f"{Path(filepath).stem}_graph.png")
        
        # Step 5: Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'class_idx': int(prediction),
                'probabilities': {name: float(prob) for name, prob in zip(class_names, probabilities)}
            },
            'metrics': {
                'num_regions': result['num_regions'],
                'num_cells': result['segmentation_metadata']['num_cells'],
                'image_shape': result['image_shape']
            },
            'features': {
                'mean_area': np.mean([f['area'] for f in result['region_features']]),
                'mean_intensity': np.mean([f['mean_intensity'] for f in result['region_features']]),
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges()
            },
            'visualizations': {
                'overlay': f"{Path(filepath).stem}_overlay.png",
                'graph': f"{Path(filepath).stem}_graph.png"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve generated output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/metrics')
def get_metrics():
    """Get model evaluation metrics"""
    try:
        metrics_path = os.path.join(app.config['OUTPUT_FOLDER'], 'models', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Metrics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)
