"""
Flask web application for protein localization analysis.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import json
import os
from werkzeug.utils import secure_filename
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = '/mnt/d/5TH_SEM/CELLULAR/input'
app.config['OUTPUT_FOLDER'] = '/mnt/d/5TH_SEM/CELLULAR/output'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
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
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file type. Only TIFF files are allowed.'}), 400


@app.route('/process', methods=['POST'])
def process_image():
    """Process uploaded image through the pipeline."""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Import processing modules
        from utils.image_preprocessing import TIFFLoader
        from segmentation.cellpose_segmentation import CellposeSegmenter
        from models.cnn_model import ProteinLocalizationCNN, CNNTrainer
        from utils.graph_construction import SuperpixelGenerator, GraphConstructor
        from models.gnn_model import create_gnn_model, GNNTrainer
        from utils.model_fusion import ModelFusion
        from utils.visualization import Visualizer
        
        # Initialize components
        loader = TIFFLoader(target_size=(224, 224))
        segmenter = CellposeSegmenter()
        visualizer = Visualizer(output_dir=app.config['OUTPUT_FOLDER'] + '/graphs')
        
        # Load and preprocess image
        original, processed = loader.preprocess(filepath)
        
        # Segment
        masks, seg_info = segmenter.segment(original)
        seg_path = os.path.join(app.config['OUTPUT_FOLDER'], 'segmented', f"{Path(filename).stem}_segment.png")
        os.makedirs(os.path.dirname(seg_path), exist_ok=True)
        segmenter.visualize_segmentation(original, masks, save_path=seg_path)
        
        # Generate superpixels and graph
        sp_gen = SuperpixelGenerator(method='slic', n_segments=100)
        segments = sp_gen.generate(original)
        features = sp_gen.extract_features(original, segments)
        
        constructor = GraphConstructor()
        graph = constructor.build_adjacency_graph(segments)
        edge_index, node_features = constructor.to_pytorch_geometric(graph, features)
        
        # Mock predictions (in real implementation, load trained models)
        import numpy as np
        cnn_probs = np.random.dirichlet(np.ones(5))
        gnn_probs = np.random.dirichlet(np.ones(5))
        
        # Fusion
        fusion = ModelFusion(method='weighted_average')
        fused_class, fused_probs = fusion.fuse(cnn_probs, gnn_probs)
        
        class_names = ['Soma', 'Dendrites', 'Axon', 'Synapses', 'Nucleus']
        
        # Create visualizations
        vis_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'graphs')
        os.makedirs(vis_dir, exist_ok=True)
        
        visualizer.plot_probability_distribution(
            cnn_probs, class_names,
            f"{Path(filename).stem}_cnn_probs.png",
            "CNN Predictions"
        )
        
        visualizer.plot_probability_distribution(
            gnn_probs, class_names,
            f"{Path(filename).stem}_gnn_probs.png",
            "GNN Predictions"
        )
        
        visualizer.plot_probability_distribution(
            fused_probs, class_names,
            f"{Path(filename).stem}_fused_probs.png",
            "Fused Predictions"
        )
        
        # Prepare response
        result = {
            'success': True,
            'filename': filename,
            'segmentation': {
                'n_regions': seg_info['n_cells'],
                'image_path': f"/output/segmented/{Path(filename).stem}_segment.png"
            },
            'predictions': {
                'cnn': {
                    'class': int(np.argmax(cnn_probs)),
                    'class_name': class_names[int(np.argmax(cnn_probs))],
                    'probabilities': cnn_probs.tolist(),
                    'plot_path': f"/output/graphs/{Path(filename).stem}_cnn_probs.png"
                },
                'gnn': {
                    'class': int(np.argmax(gnn_probs)),
                    'class_name': class_names[int(np.argmax(gnn_probs))],
                    'probabilities': gnn_probs.tolist(),
                    'plot_path': f"/output/graphs/{Path(filename).stem}_gnn_probs.png"
                },
                'fused': {
                    'class': int(fused_class),
                    'class_name': class_names[int(fused_class)],
                    'probabilities': fused_probs.tolist(),
                    'plot_path': f"/output/graphs/{Path(filename).stem}_fused_probs.png"
                }
            },
            'graph_info': {
                'n_nodes': graph.number_of_nodes(),
                'n_edges': graph.number_of_edges()
            }
        }
        
        # Save result as JSON
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], 'reports', f"{Path(filename).stem}_result.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/batch-process', methods=['POST'])
def batch_process():
    """Process all images in the input directory."""
    try:
        input_dir = app.config['UPLOAD_FOLDER']
        tiff_files = []
        
        for ext in ['*.tif', '*.tiff']:
            tiff_files.extend(Path(input_dir).rglob(ext))
        
        if not tiff_files:
            return jsonify({'error': 'No TIFF files found in input directory'}), 404
        
        results = []
        for filepath in tiff_files:
            # Process each file (simplified)
            filename = filepath.name
            result = {
                'filename': filename,
                'status': 'processed',
                'path': str(filepath)
            }
            results.append(result)
        
        return jsonify({
            'success': True,
            'total_files': len(tiff_files),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve output files."""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'File not found'}), 404


@app.route('/list-files', methods=['GET'])
def list_files():
    """List all processed files."""
    try:
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        files = {
            'segmented': [f.name for f in (output_dir / 'segmented').glob('*.png')] if (output_dir / 'segmented').exists() else [],
            'graphs': [f.name for f in (output_dir / 'graphs').glob('*.png')] if (output_dir / 'graphs').exists() else [],
            'reports': [f.name for f in (output_dir / 'reports').glob('*.json')] if (output_dir / 'reports').exists() else []
        }
        
        return jsonify({
            'success': True,
            'files': files
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
