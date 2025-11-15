"""
Web interface for protein localization pipeline using Gradio
Allows unrestricted TIFF file uploads and end-to-end processing
"""
import gradio as gr
import numpy as np
import torch
from pathlib import Path
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.segmentation import TIFFLoader, CellposeSegmenter
from preprocessing.feature_extraction import FeatureExtractor
from graph_construction.graph_builder import GraphConstructor, PyTorchGeometricConverter
from visualization.plotters import SegmentationVisualizer, StatisticalPlotter
from visualization.graph_viz import GraphVisualizer
from visualization.metrics import MetricsEvaluator


class ProteinLocalizationInterface:
    """Interface for protein localization pipeline"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize interface
        
        Args:
            model_path: Path to trained model (optional)
        """
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize components
        self.loader = TIFFLoader()
        self.segmenter = CellposeSegmenter()
        self.feature_extractor = FeatureExtractor()
        self.graph_constructor = GraphConstructor()
        
        # Create temp directory for visualizations
        self.viz_dir = tempfile.mkdtemp()
        self.seg_viz = SegmentationVisualizer(output_dir=self.viz_dir)
        self.graph_viz = GraphVisualizer(output_dir=self.viz_dir)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Model loading logic here
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def process_tiff(self, tiff_file):
        """
        Process uploaded TIFF file through the entire pipeline
        
        Args:
            tiff_file: Uploaded TIFF file
        
        Returns:
            Tuple of (results_text, seg_image, graph_image, metrics_dict)
        """
        try:
            # Load TIFF
            image = self.loader.load_tiff(tiff_file.name)
            if image is None:
                return "Error: Could not load TIFF file", None, None, {}
            
            results = []
            results.append(f"✓ Loaded TIFF image: {image.shape}")
            
            # Segmentation
            masks, seg_info = self.segmenter.segment_image(image)
            if masks is None:
                return "Error: Segmentation failed", None, None, {}
            
            results.append(f"✓ Segmentation complete: {seg_info['num_cells']} cells detected")
            
            # Feature extraction
            features = self.feature_extractor.extract_all_features(image, masks)
            if features.empty:
                return "Error: Feature extraction failed", None, None, {}
            
            results.append(f"✓ Extracted {len(features.columns)} features from {len(features)} regions")
            
            # Graph construction
            graph = self.graph_constructor.construct_graph(features, masks)
            results.append(f"✓ Constructed graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Create visualizations
            seg_filename = "segmentation.png"
            self.seg_viz.plot_segmentation_overlay(
                image, masks,
                title="Segmentation Result",
                filename=seg_filename
            )
            seg_path = os.path.join(self.viz_dir, seg_filename)
            
            graph_filename = "graph.png"
            self.graph_viz.plot_graph(
                graph,
                title="Protein Localization Graph",
                filename=graph_filename
            )
            graph_path = os.path.join(self.viz_dir, graph_filename)
            
            # Prediction (if model is loaded)
            prediction = "No model loaded"
            confidence = 0.0
            
            if self.model is not None:
                # Run inference
                # This would use the actual model
                prediction = "Predicted Class: Example"
                confidence = 0.95
            
            results.append(f"\n📊 Prediction: {prediction}")
            results.append(f"📊 Confidence: {confidence:.2%}")
            
            # Prepare metrics
            metrics = {
                'num_cells': seg_info['num_cells'],
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'avg_degree': 2 * graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
                'prediction': prediction,
                'confidence': confidence
            }
            
            # Format results
            results_text = "\n".join(results)
            results_text += "\n\n📈 Graph Statistics:"
            results_text += f"\n  - Average degree: {metrics['avg_degree']:.2f}"
            results_text += f"\n  - Graph density: {graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes()-1)/2) if graph.number_of_nodes() > 1 else 0:.4f}"
            
            return results_text, seg_path, graph_path, metrics
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
            return error_msg, None, None, {}
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="Protein Sub-Cellular Localization", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🧬 Protein Sub-Cellular Localization Pipeline
            
            Upload any TIFF microscopy image (no size restrictions) for automatic segmentation, 
            feature extraction, graph construction, and localization prediction.
            
            ### Pipeline Steps:
            1. **Segmentation**: Detect neuronal structures using Cellpose
            2. **Feature Extraction**: Extract spatial, morphological, and intensity features
            3. **Graph Construction**: Build graph representation with nodes and edges
            4. **Prediction**: Classify protein localization (requires trained model)
            """)
            
            with gr.Row():
                with gr.Column():
                    # Input
                    tiff_input = gr.File(
                        label="Upload TIFF Image",
                        file_types=[".tif", ".tiff"],
                        type="filepath"
                    )
                    
                    process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    **Supported formats**: .tif, .tiff
                    
                    **No file size restrictions** - Upload files of any size
                    """)
                
                with gr.Column():
                    # Results
                    results_text = gr.Textbox(
                        label="Processing Results",
                        lines=15,
                        max_lines=20
                    )
            
            with gr.Row():
                # Visualizations
                seg_output = gr.Image(label="Segmentation Overlay", type="filepath")
                graph_output = gr.Image(label="Graph Visualization", type="filepath")
            
            with gr.Row():
                # Metrics
                metrics_json = gr.JSON(label="Detailed Metrics")
            
            # Event handlers
            process_btn.click(
                fn=self.process_tiff,
                inputs=[tiff_input],
                outputs=[results_text, seg_output, graph_output, metrics_json]
            )
            
            gr.Markdown("""
            ---
            ### About This Pipeline
            
            This pipeline implements a complete workflow for analyzing 4D TIFF microscopy images:
            
            - **Preprocessing**: Recursive directory scanning and TIFF loading
            - **Segmentation**: Cellpose-based detection of cellular structures
            - **Feature Extraction**: Spatial, morphological, intensity, and region-level features
            - **Graph Construction**: Biologically meaningful graph with spatial relationships
            - **Model Inference**: Graph-CNN and VGG-16 based classification
            - **Visualization**: Publication-ready plots and graphs
            
            All processed data and models are saved to: `/mnt/d/5TH_SEM/CELLULAR/output/output`
            """)
        
        return interface


def launch_interface(model_path: str = None, share: bool = False):
    """
    Launch the Gradio interface
    
    Args:
        model_path: Path to trained model
        share: Whether to create public link
    """
    app = ProteinLocalizationInterface(model_path=model_path)
    interface = app.create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        inbrowser=True
    )


if __name__ == "__main__":
    print("Starting Protein Localization Interface...")
    launch_interface(share=False)
