"""
Flask web application for protein localization prediction.
Provides file upload interface and real-time prediction results.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from pathlib import Path
import numpy as np
import torch
from werkzeug.utils import secure_filename
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import TIFFPreprocessor
from graph_builder import BiologicalGraphBuilder
from models import ModelTrainer
from visualization import ProteinVisualization

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/mnt/d/5TH_SEM/CELLULAR/output'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

# Global variables for loaded model
model_trainer = None
model_loaded = False


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'tif', 'tiff'}


def process_uploaded_file(file_path):
    """
    Process uploaded TIFF file through the complete pipeline.
    
    Args:
        file_path: Path to uploaded file
        
    Returns:
        Dictionary with results
    """
    try:
        # Initialize components
        preprocessor = TIFFPreprocessor(
            input_dir=str(Path(file_path).parent),
            output_dir=app.config['OUTPUT_FOLDER']
        )
        
        # Process TIFF
        print("Processing TIFF...")
        result = preprocessor.process_single_tiff(Path(file_path))
        
        if result is None:
            return {'error': 'Failed to process TIFF file'}
        
        # Build graph
        print("Building graph...")
        graph_builder = BiologicalGraphBuilder()
        G = graph_builder.build_graph(result['features'], method='knn')
        pyg_data = graph_builder.networkx_to_pyg(G)
        
        # Create visualizations
        print("Creating visualizations...")
        viz = ProteinVisualization(app.config['OUTPUT_FOLDER'])
        
        # Save graph visualization
        base_name = Path(file_path).stem
        graph_file = f"{base_name}_graph.png"
        viz.plot_graph(G, filename=graph_file)
        
        # Predict (if model is loaded)
        prediction = None
        accuracy = None
        metrics = {}
        
        if model_loaded and model_trainer is not None:
            print("Running prediction...")
            try:
                model_trainer.model.eval()
                with torch.no_grad():
                    pyg_data = pyg_data.to(model_trainer.device)
                    output = model_trainer.model(pyg_data.unsqueeze(0))
                    pred = output.argmax(dim=1).item()
                    prediction = f"Class {pred}"
                    
                    # Calculate confidence
                    probs = torch.exp(output)
                    confidence = probs[0, pred].item()
                    accuracy = confidence * 100
                    
                    metrics = {
                        'predicted_class': pred,
                        'confidence': confidence,
                        'accuracy': accuracy
                    }
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction = "Prediction unavailable"
        
        # Prepare response
        response = {
            'success': True,
            'file_name': Path(file_path).name,
            'n_regions': result['n_regions'],
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'graph_visualization': graph_file,
            'prediction': prediction or 'Model not loaded',
            'accuracy': f"{accuracy:.2f}%" if accuracy else 'N/A',
            'metrics': metrics,
            'segmentation_info': result['segmentation_metadata']
        }
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only TIFF files allowed.'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Process file
    result = process_uploaded_file(filepath)
    
    # Clean up uploaded file
    try:
        os.remove(filepath)
    except:
        pass
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify(result)


@app.route('/results/<filename>')
def get_result_file(filename):
    """Serve result files."""
    file_path = Path(app.config['OUTPUT_FOLDER']) / filename
    if file_path.exists():
        return send_file(str(file_path))
    return "File not found", 404


@app.route('/load_model', methods=['POST'])
def load_model():
    """Load trained model."""
    global model_trainer, model_loaded
    
    try:
        model_path = Path(app.config['OUTPUT_FOLDER']) / 'models' / 'graph_cnn_model.pt'
        
        if not model_path.exists():
            return jsonify({'error': 'Model file not found'}), 404
        
        model_trainer = ModelTrainer(model_type='graph_cnn', num_classes=5)
        model_trainer.load_model(str(model_path), num_node_features=10)
        model_loaded = True
        
        return jsonify({'success': True, 'message': 'Model loaded successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Get system status."""
    return jsonify({
        'model_loaded': model_loaded,
        'output_dir': app.config['OUTPUT_FOLDER'],
        'status': 'ready'
    })


if __name__ == '__main__':
    print("Starting Protein Localization Web Interface...")
    print(f"Output directory: {app.config['OUTPUT_FOLDER']}")
    app.run(host='0.0.0.0', port=5000, debug=True)
