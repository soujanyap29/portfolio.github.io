"""
Flask web application for Protein Sub-Cellular Localization Analysis
"""
from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import sys
import json
from werkzeug.utils import secure_filename
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from pipeline import ProteinLocalizationPipeline
import config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize pipeline
pipeline = ProteinLocalizationPipeline()

ALLOWED_EXTENSIONS = {'tif', 'tiff'}


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                          project_name="Protein Sub-Cellular Localization in Neurons")


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle single file upload and analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only TIFF files (.tif, .tiff) are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze image
        result = pipeline.analyze_single_image(filepath, save_results=True)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Clean up uploaded file
        # os.remove(filepath)  # Keep for now for debugging
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_process', methods=['POST'])
def batch_process():
    """Handle batch processing of directory"""
    data = request.get_json()
    
    if not data or 'directory' not in data:
        # Use default directory
        directory = config.INPUT_DIR
    else:
        directory = data['directory']
    
    if not os.path.exists(directory):
        return jsonify({
            'error': f'Directory not found: {directory}',
            'note': 'Using default directory from config'
        }), 400
    
    try:
        results = pipeline.batch_process(directory)
        return jsonify({
            'status': 'success',
            'total_processed': len(results),
            'results': results
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results/<filename>')
def get_result(filename):
    """Retrieve analysis result"""
    filepath = os.path.join(config.REPORTS_DIR, f"{filename}_report.json")
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Result not found'}), 404
    
    with open(filepath, 'r') as f:
        result = json.load(f)
    
    return jsonify(result), 200


@app.route('/download/<result_type>/<filename>')
def download_file(result_type, filename):
    """Download result files"""
    if result_type == 'segmentation':
        directory = config.SEGMENTED_DIR
    elif result_type == 'graph':
        directory = config.GRAPHS_DIR
    elif result_type == 'report':
        directory = config.REPORTS_DIR
    else:
        return jsonify({'error': 'Invalid result type'}), 400
    
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, as_attachment=True)


@app.route('/batch_summary')
def batch_summary():
    """Get batch processing summary"""
    summary_path = os.path.join(config.REPORTS_DIR, 'batch_summary.json')
    
    if not os.path.exists(summary_path):
        return jsonify({'error': 'No batch summary available'}), 404
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return jsonify(summary), 200


@app.route('/available_results')
def available_results():
    """List all available analysis results"""
    results = []
    
    if os.path.exists(config.REPORTS_DIR):
        for filename in os.listdir(config.REPORTS_DIR):
            if filename.endswith('_report.json'):
                filepath = os.path.join(config.REPORTS_DIR, filename)
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    results.append({
                        'filename': filename,
                        'image_name': result.get('image_name', 'Unknown'),
                        'timestamp': result.get('timestamp', 'Unknown'),
                        'prediction': result.get('fused_prediction', {})
                    })
    
    return jsonify({'results': results}), 200


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pipeline_initialized': True,
        'output_dir': config.OUTPUT_DIR
    }), 200


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PROTEIN SUB-CELLULAR LOCALIZATION ANALYSIS SYSTEM")
    print("="*70)
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print(f"Input Directory: {config.INPUT_DIR}")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
