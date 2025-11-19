"""
Main inference pipeline for protein localization
"""
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Tuple, List
import sys

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from config import *
from image_loader import TIFFLoader
from segmentation import SegmentationModule, save_segmentation
from cnn_model import VGG16Classifier
from gnn_model import GraphConstructor, GNNClassifier
from model_fusion import ModelFusion
from evaluation import EvaluationMetrics, compute_colocalization_metrics
from visualization import ScientificVisualizer


class ProteinLocalizationPipeline:
    """Complete pipeline for protein sub-cellular localization"""
    
    def __init__(self, output_dir: str = OUTPUT_PATH):
        """
        Initialize pipeline
        
        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        self.graphs_dir = os.path.join(output_dir, "graphs")
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "segmented"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "reports"), exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        # Initialize components
        self.tiff_loader = TIFFLoader()
        self.segmentation_module = SegmentationModule(method=SEGMENTATION_METHOD)
        self.cnn_classifier = VGG16Classifier(num_classes=len(PROTEIN_CLASSES))
        self.gnn_classifier = GNNClassifier(model_type="GCN", num_classes=len(PROTEIN_CLASSES))
        self.visualizer = ScientificVisualizer(dpi=DPI)
        
        print("✓ Pipeline initialized successfully")
    
    def process_single_image(self, image_path: str) -> Dict:
        """
        Process a single TIFF image
        
        Args:
            image_path: Path to TIFF image
            
        Returns:
            Dictionary containing all results
        """
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        # Load image
        print("1. Loading TIFF image...")
        image = self.tiff_loader.load_tiff(image_path)
        if image is None:
            return {"error": "Failed to load image"}
        
        image_normalized = self.tiff_loader.normalize_image(image)
        
        # Segmentation
        print("2. Performing segmentation...")
        segments = self.segmentation_module.segment(
            image_normalized,
            n_segments=SLIC_N_SEGMENTS,
            compactness=SLIC_COMPACTNESS
        )
        
        # Save segmentation
        filename = os.path.splitext(os.path.basename(image_path))[0]
        seg_path = os.path.join(self.results_dir, "segmented", f"{filename}_segment.png")
        save_segmentation(image_normalized, segments, seg_path)
        print(f"   Segmentation saved: {seg_path}")
        
        # Prepare image for CNN
        print("3. Running CNN classification...")
        image_for_cnn = self.tiff_loader.preprocess_for_model(image, size=IMAGE_SIZE)
        cnn_class, cnn_probs = self.cnn_classifier.predict(image_for_cnn[0])
        print(f"   CNN Prediction: {PROTEIN_CLASSES[cnn_class]} (confidence: {cnn_probs[cnn_class]:.3f})")
        
        # Build graph for GNN
        print("4. Building graph and running GNN classification...")
        graph_constructor = GraphConstructor()
        features = graph_constructor.extract_superpixel_features(image_normalized, segments)
        adjacency = graph_constructor.build_adjacency(segments, k_neighbors=5)
        graph_data = graph_constructor.create_graph_data(features, adjacency)
        
        gnn_class, gnn_probs = self.gnn_classifier.predict(graph_data)
        print(f"   GNN Prediction: {PROTEIN_CLASSES[gnn_class]} (confidence: {gnn_probs[gnn_class]:.3f})")
        
        # Model fusion
        print("5. Fusing model predictions...")
        fused_class, fused_probs = ModelFusion.late_fusion_weighted(
            cnn_probs, gnn_probs, cnn_weight=0.6, gnn_weight=0.4
        )
        print(f"   Fused Prediction: {PROTEIN_CLASSES[fused_class]} (confidence: {fused_probs[fused_class]:.3f})")
        
        # Generate visualizations
        print("6. Generating visualizations...")
        self._generate_visualizations(
            image_normalized, segments, fused_probs, fused_class,
            adjacency, features, filename
        )
        
        # Prepare results
        results = {
            "filename": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "segmentation_path": seg_path,
            "cnn": {
                "predicted_class": PROTEIN_CLASSES[cnn_class],
                "predicted_class_index": int(cnn_class),
                "confidence": float(cnn_probs[cnn_class]),
                "probabilities": {PROTEIN_CLASSES[i]: float(cnn_probs[i]) 
                                 for i in range(len(PROTEIN_CLASSES))}
            },
            "gnn": {
                "predicted_class": PROTEIN_CLASSES[gnn_class],
                "predicted_class_index": int(gnn_class),
                "confidence": float(gnn_probs[gnn_class]),
                "probabilities": {PROTEIN_CLASSES[i]: float(gnn_probs[i])
                                 for i in range(len(PROTEIN_CLASSES))}
            },
            "fused": {
                "predicted_class": PROTEIN_CLASSES[fused_class],
                "predicted_class_index": int(fused_class),
                "confidence": float(fused_probs[fused_class]),
                "probabilities": {PROTEIN_CLASSES[i]: float(fused_probs[i])
                                 for i in range(len(PROTEIN_CLASSES))}
            },
            "visualizations": {
                "segmentation": seg_path,
                "overlay": os.path.join(self.graphs_dir, f"{filename}_overlay.png"),
                "probability": os.path.join(self.graphs_dir, f"{filename}_probabilities.png"),
                "graph": os.path.join(self.graphs_dir, f"{filename}_graph.png")
            }
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, "reports", f"{filename}_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Processing complete. Report saved: {report_path}")
        
        return results
    
    def _generate_visualizations(self, image: np.ndarray, segments: np.ndarray,
                                probabilities: np.ndarray, predicted_class: int,
                                adjacency: np.ndarray, features: np.ndarray,
                                filename: str):
        """Generate all visualizations for a sample"""
        
        # Overlay visualization
        overlay_path = os.path.join(self.graphs_dir, f"{filename}_overlay.png")
        self.visualizer.plot_image_overlay(image, segments, overlay_path,
                                          title="Protein Localization Analysis")
        
        # Probability distribution
        prob_path = os.path.join(self.graphs_dir, f"{filename}_probabilities.png")
        EvaluationMetrics.plot_probability_distribution(
            probabilities, PROTEIN_CLASSES, prob_path, predicted_class
        )
        
        # Graph visualization
        graph_path = os.path.join(self.graphs_dir, f"{filename}_graph.png")
        self.visualizer.plot_graph_visualization(
            adjacency, features, graph_path,
            title="Superpixel Graph Network"
        )
        
        # Compartment map
        compartment_path = os.path.join(self.graphs_dir, f"{filename}_compartments.png")
        self.visualizer.plot_compartment_map(segments, compartment_path)
    
    def process_batch(self, input_dir: str = INPUT_PATH) -> List[Dict]:
        """
        Process all TIFF files in a directory (recursive)
        
        Args:
            input_dir: Input directory path
            
        Returns:
            List of results for all images
        """
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING")
        print(f"Input Directory: {input_dir}")
        print(f"{'='*80}\n")
        
        # Scan for TIFF files
        tiff_files = self.tiff_loader.scan_directory(input_dir, recursive=True)
        
        if not tiff_files:
            print(f"No TIFF files found in {input_dir}")
            return []
        
        print(f"Found {len(tiff_files)} TIFF files")
        
        # Process each file
        all_results = []
        for i, tiff_file in enumerate(tiff_files, 1):
            print(f"\n[{i}/{len(tiff_files)}] Processing {tiff_file}")
            try:
                result = self.process_single_image(tiff_file)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {tiff_file}: {e}")
                all_results.append({"filename": os.path.basename(tiff_file), "error": str(e)})
        
        # Save batch summary
        summary_path = os.path.join(self.results_dir, "reports", "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "total_images": len(tiff_files),
                "successful": len([r for r in all_results if "error" not in r]),
                "failed": len([r for r in all_results if "error" in r]),
                "results": all_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=4)
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"Summary saved: {summary_path}")
        print(f"{'='*80}\n")
        
        return all_results


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Protein Sub-Cellular Localization System")
    parser.add_argument("--image", type=str, help="Path to single TIFF image")
    parser.add_argument("--batch", type=str, help="Path to directory for batch processing")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProteinLocalizationPipeline(output_dir=args.output)
    
    if args.image:
        # Process single image
        result = pipeline.process_single_image(args.image)
        print("\nResults:")
        print(json.dumps(result, indent=2))
    
    elif args.batch:
        # Process batch
        results = pipeline.process_batch(input_dir=args.batch)
        print(f"\nProcessed {len(results)} images")
    
    else:
        print("Please provide either --image or --batch argument")
        parser.print_help()


if __name__ == "__main__":
    main()
