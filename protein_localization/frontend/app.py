"""
Flask web interface for protein localization prediction
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
from pathlib import Path
import numpy as np
import io
import base64
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing import TIFFProcessor
from graph_construction import BiologicalGraphBuilder
from visualization import ScientificVisualizer
from utils import load_config, ensure_dir

app = Flask(__name__)
CORS(app)

# Load configuration
config = load_config()
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = config['data']['output_dir']
ensure_dir(UPLOAD_FOLDER)
ensure_dir(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# Initialize processors
processor = TIFFProcessor(config)
graph_builder = BiologicalGraphBuilder(config)
visualizer = ScientificVisualizer(config)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle TIFF file upload and process it through the pipeline
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.tif', '.tiff')):
        return jsonify({'error': 'Invalid file format. Only TIFF files allowed'}), 400
    
    try:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process through pipeline
        result = process_pipeline(filepath, file.filename)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_pipeline(filepath: str, filename: str):
    """
    Complete processing pipeline
    
    Args:
        filepath: Path to uploaded TIFF file
        filename: Original filename
        
    Returns:
        Dictionary with results
    """
    base_name = Path(filename).stem
    
    # Step 1: Preprocessing
    img, masks, features = processor.process_single_tiff(filepath)
    
    # Step 2: Graph construction
    G = graph_builder.build_graph(features)
    
    # Step 3: Generate visualizations
    output_dir = os.path.join(OUTPUT_FOLDER, base_name)
    ensure_dir(output_dir)
    
    seg_path = os.path.join(output_dir, 'segmentation_overlay.png')
    visualizer.plot_segmentation_overlay(img, masks, seg_path)
    
    graph_path = os.path.join(output_dir, 'graph_visualization.png')
    visualizer.plot_graph_visualization(G, graph_path)
    
    comp_path = os.path.join(output_dir, 'compartment_map.png')
    visualizer.plot_compartment_map(masks, comp_path)
    
    # Step 4: Model prediction (placeholder - needs trained model)
    prediction_class = "Mitochondrial"  # Placeholder
    prediction_confidence = 0.85  # Placeholder
    
    # Step 5: Calculate metrics
    num_regions = len(features['region_ids'])
    avg_area = np.mean(features['areas'])
    avg_intensity = np.mean(features['mean_intensities'])
    
    # Convert images to base64 for web display
    seg_img_b64 = image_to_base64(seg_path)
    graph_img_b64 = image_to_base64(graph_path)
    comp_img_b64 = image_to_base64(comp_path)
    
    result = {
        'filename': filename,
        'prediction': {
            'class': prediction_class,
            'confidence': prediction_confidence
        },
        'metrics': {
            'num_regions': num_regions,
            'avg_area': float(avg_area),
            'avg_intensity': float(avg_intensity),
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        },
        'visualizations': {
            'segmentation': seg_img_b64,
            'graph': graph_img_b64,
            'compartment': comp_img_b64
        },
        'output_directory': output_dir
    }
    
    return result


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    host = config['frontend']['host']
    port = config['frontend']['port']
    app.run(host=host, port=port, debug=True)
