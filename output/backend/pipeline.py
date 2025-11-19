"""
Main pipeline orchestrator for protein localization analysis
"""
import os
import json
import numpy as np
from datetime import datetime

from image_processor import ImageProcessor
from segmentation import SegmentationEngine
from cnn_classifier import CNNPredictor
from gnn_classifier import GNNPredictor
from evaluation import ModelFusion, MetricsCalculator
from visualization import ScientificVisualizer
import config


class ProteinLocalizationPipeline:
    """Main pipeline for analyzing protein sub-cellular localization"""
    
    def __init__(self):
        """Initialize all components"""
        self.image_processor = ImageProcessor(target_size=config.IMAGE_SIZE)
        self.segmentation_engine = SegmentationEngine(method='slic')
        self.cnn_predictor = CNNPredictor(num_classes=config.NUM_CLASSES)
        self.gnn_predictor = GNNPredictor(num_classes=config.NUM_CLASSES)
        self.model_fusion = ModelFusion(
            cnn_weight=config.VGG16_WEIGHT,
            gnn_weight=config.GNN_WEIGHT
        )
        self.metrics_calculator = MetricsCalculator(config.LOCALIZATION_CLASSES)
        self.visualizer = ScientificVisualizer(dpi=config.DPI)
    
    def analyze_single_image(self, image_path, save_results=True):
        """
        Analyze a single TIFF image
        
        Args:
            image_path: Path to TIFF file
            save_results: Whether to save outputs
            
        Returns:
            Dictionary containing all results
        """
        print(f"\nAnalyzing image: {os.path.basename(image_path)}")
        
        # 1. Load and preprocess image
        print("  [1/6] Loading and preprocessing...")
        original_image = self.image_processor.load_tiff(image_path)
        if original_image is None:
            return {'error': 'Failed to load image'}
        
        preprocessed_image = self.image_processor.preprocess(image_path)
        
        # 2. Segmentation
        print("  [2/6] Performing segmentation...")
        segments = self.segmentation_engine.segment(
            preprocessed_image,
            n_segments=config.SLIC_N_SEGMENTS,
            compactness=config.SLIC_COMPACTNESS,
            sigma=config.SLIC_SIGMA
        )
        
        # 3. CNN prediction
        print("  [3/6] Running CNN classifier...")
        cnn_probs, cnn_class, cnn_conf = self.cnn_predictor.predict(preprocessed_image)
        
        # 4. GNN prediction
        print("  [4/6] Running GNN classifier...")
        gnn_probs, gnn_class, gnn_conf = self.gnn_predictor.predict(
            preprocessed_image, segments
        )
        
        # 5. Model fusion
        print("  [5/6] Fusing model predictions...")
        fused_probs, final_class, final_conf = self.model_fusion.fuse_predictions(
            cnn_probs, gnn_probs
        )
        
        # 6. Generate visualizations and reports
        print("  [6/6] Generating visualizations and reports...")
        results = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'timestamp': datetime.now().isoformat(),
            'cnn_prediction': {
                'class': config.LOCALIZATION_CLASSES[cnn_class],
                'class_index': int(cnn_class),
                'confidence': float(cnn_conf),
                'probabilities': {
                    cls: float(prob) 
                    for cls, prob in zip(config.LOCALIZATION_CLASSES, cnn_probs)
                }
            },
            'gnn_prediction': {
                'class': config.LOCALIZATION_CLASSES[gnn_class],
                'class_index': int(gnn_class),
                'confidence': float(gnn_conf),
                'probabilities': {
                    cls: float(prob)
                    for cls, prob in zip(config.LOCALIZATION_CLASSES, gnn_probs)
                }
            },
            'fused_prediction': {
                'class': config.LOCALIZATION_CLASSES[final_class],
                'class_index': int(final_class),
                'confidence': float(final_conf),
                'probabilities': {
                    cls: float(prob)
                    for cls, prob in zip(config.LOCALIZATION_CLASSES, fused_probs)
                }
            }
        }
        
        if save_results:
            self._save_results(
                image_path, original_image, preprocessed_image, 
                segments, results
            )
        
        print("  ✓ Analysis complete!")
        return results
    
    def _save_results(self, image_path, original_image, preprocessed_image, 
                     segments, results):
        """Save all outputs to disk"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save segmentation
        seg_path = os.path.join(config.SEGMENTED_DIR, f"{base_name}_segment.png")
        self.visualizer.plot_image_with_segmentation(
            preprocessed_image, segments, seg_path,
            title=f"Segmentation: {base_name}"
        )
        
        # Save probability distributions
        prob_path = os.path.join(config.GRAPHS_DIR, f"{base_name}_probabilities.png")
        self.visualizer.plot_probability_distribution(
            results['fused_prediction']['probabilities'].values(),
            config.LOCALIZATION_CLASSES,
            prob_path,
            title=f"Prediction Probabilities: {base_name}"
        )
        
        # Save intensity profile
        intensity_path = os.path.join(config.GRAPHS_DIR, f"{base_name}_intensity.png")
        self.visualizer.plot_intensity_profile(
            preprocessed_image, intensity_path,
            title=f"Intensity Profile: {base_name}"
        )
        
        # Save JSON report
        report_path = os.path.join(config.REPORTS_DIR, f"{base_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        results['saved_files'] = {
            'segmentation': seg_path,
            'probabilities': prob_path,
            'intensity_profile': intensity_path,
            'report': report_path
        }
    
    def batch_process(self, input_directory=None):
        """
        Process all TIFF images in a directory recursively
        
        Args:
            input_directory: Root directory to scan (defaults to config.INPUT_DIR)
            
        Returns:
            List of results for all images
        """
        if input_directory is None:
            input_directory = config.INPUT_DIR
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING")
        print(f"{'='*70}")
        print(f"Scanning directory: {input_directory}")
        
        # Scan for TIFF files
        tiff_files = self.image_processor.scan_directory(input_directory)
        print(f"Found {len(tiff_files)} TIFF files")
        
        if len(tiff_files) == 0:
            print("No TIFF files found!")
            return []
        
        # Process each file
        all_results = []
        for i, tiff_file in enumerate(tiff_files, 1):
            print(f"\n[{i}/{len(tiff_files)}] Processing: {os.path.basename(tiff_file)}")
            result = self.analyze_single_image(tiff_file, save_results=True)
            all_results.append(result)
        
        # Generate batch summary
        self._generate_batch_summary(all_results)
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total images processed: {len(all_results)}")
        print(f"Results saved to: {config.OUTPUT_DIR}")
        
        return all_results
    
    def _generate_batch_summary(self, results):
        """Generate summary visualizations for batch processing"""
        print("\nGenerating batch summary...")
        
        # Count predictions by class
        class_counts = {cls: 0 for cls in config.LOCALIZATION_CLASSES}
        for result in results:
            if 'error' not in result:
                pred_class = result['fused_prediction']['class']
                class_counts[pred_class] += 1
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'class_distribution': class_counts,
            'results': results
        }
        
        summary_path = os.path.join(config.REPORTS_DIR, 'batch_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Batch summary saved to: {summary_path}")
    
    def evaluate_model(self, test_data_path, labels_path):
        """
        Evaluate model on test dataset with ground truth labels
        
        Args:
            test_data_path: Directory containing test images
            labels_path: Path to JSON file with ground truth labels
            
        Returns:
            Evaluation metrics
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Load ground truth labels
        with open(labels_path, 'r') as f:
            ground_truth = json.load(f)
        
        # Process test images
        predictions = []
        true_labels = []
        
        for image_name, true_label in ground_truth.items():
            image_path = os.path.join(test_data_path, image_name)
            if os.path.exists(image_path):
                result = self.analyze_single_image(image_path, save_results=False)
                if 'error' not in result:
                    predictions.append(result['fused_prediction']['class_index'])
                    true_labels.append(true_label)
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        metrics = self.metrics_calculator.calculate_metrics(true_labels, predictions)
        
        # Generate visualizations
        self._visualize_evaluation(metrics, true_labels, predictions)
        
        # Save evaluation report
        eval_report_path = os.path.join(config.REPORTS_DIR, 'evaluation_report.json')
        self.metrics_calculator.save_report(metrics, eval_report_path)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        return metrics
    
    def _visualize_evaluation(self, metrics, y_true, y_pred):
        """Generate evaluation visualizations"""
        # Confusion matrix
        cm_path = os.path.join(config.GRAPHS_DIR, 'confusion_matrix.png')
        self.visualizer.plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            config.LOCALIZATION_CLASSES,
            cm_path
        )
        
        # Overall metrics
        metrics_path = os.path.join(config.GRAPHS_DIR, 'overall_metrics.png')
        self.visualizer.plot_metrics_comparison(metrics, metrics_path)
        
        # Per-class metrics
        if 'per_class' in metrics:
            per_class_path = os.path.join(config.GRAPHS_DIR, 'per_class_metrics.png')
            self.visualizer.plot_per_class_metrics(
                metrics['per_class'],
                config.LOCALIZATION_CLASSES,
                per_class_path
            )


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ProteinLocalizationPipeline()
    
    # Example: Analyze single image
    # result = pipeline.analyze_single_image("path/to/image.tif")
    
    # Example: Batch process directory
    # results = pipeline.batch_process()
    
    print("\nProtein Localization Pipeline initialized successfully!")
    print("Use pipeline.analyze_single_image(path) or pipeline.batch_process() to start analysis.")
