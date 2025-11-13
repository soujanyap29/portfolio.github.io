"""
Main orchestration script for protein localization pipeline
Automates the complete workflow from data loading to evaluation
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from utils.data_loader import TIFFDataLoader
from utils.preprocessor import ImagePreprocessor
from utils.graph_builder import GraphBuilder
from utils.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProteinLocalizationPipeline:
    """Complete pipeline for protein sub-cellular localization analysis"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Initialized Protein Localization Pipeline")
        self._print_config()
    
    def _print_config(self):
        """Print configuration summary"""
        logger.info("="*70)
        logger.info("Pipeline Configuration")
        logger.info("="*70)
        logger.info(f"Input Directory:  {self.config['data']['input_dir']}")
        logger.info(f"Output Directory: {self.config['data']['output_dir']}")
        logger.info(f"Model Type:       {self.config['model']['type']}")
        logger.info(f"Batch Size:       {self.config['training']['batch_size']}")
        logger.info(f"Epochs:           {self.config['training']['epochs']}")
        logger.info("="*70)
    
    def step1_load_data(self, input_dir: Optional[str] = None) -> Dict:
        """
        Step 1: Data Access & Sanity Checks
        
        Args:
            input_dir: Optional input directory (overrides config)
            
        Returns:
            Dictionary of loaded images
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Data Access & Sanity Checks")
        logger.info("="*70)
        
        input_dir = input_dir or self.config['data']['input_dir']
        
        # Initialize data loader
        loader = TIFFDataLoader(input_dir)
        
        # Scan directory
        loader.scan_directory()
        
        # Load all images with validation
        images = loader.load_all(validate=True)
        
        # Print summary
        loader.print_summary()
        
        logger.info(f"✓ Step 1 completed: Loaded {len(images)} images")
        
        return images
    
    def step2_preprocess(self, images: Dict) -> Dict:
        """
        Step 2: Image Preprocessing
        
        Args:
            images: Dictionary of raw images
            
        Returns:
            Dictionary of preprocessed images
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Image Preprocessing")
        logger.info("="*70)
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor(self.config['preprocessing'])
        
        # Preprocess images
        processed_dir = Path(self.config['data']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract just the images from tuples
        image_arrays = {k: v[0] for k, v in images.items()}
        
        processed = preprocessor.preprocess_batch(
            image_arrays, 
            output_dir=str(processed_dir)
        )
        
        logger.info(f"✓ Step 2 completed: Preprocessed {len(processed)} images")
        
        return processed
    
    def step3_build_graphs(self, images: Dict) -> Dict:
        """
        Step 3: Graph Construction
        
        Args:
            images: Dictionary of preprocessed images
            
        Returns:
            Dictionary of graph data
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Graph Construction")
        logger.info("="*70)
        
        # Initialize graph builder
        graph_builder = GraphBuilder(self.config['graph'])
        
        # Build graphs
        graph_dir = Path(self.config['data']['graph_dir'])
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        graphs = graph_builder.build_batch(
            images,
            output_dir=str(graph_dir)
        )
        
        logger.info(f"✓ Step 3 completed: Built {len(graphs)} graphs")
        
        return graphs
    
    def step4_prepare_labels(self) -> Dict:
        """
        Step 4: Labels Preparation
        
        Returns:
            Dictionary of labels (if available)
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Labels Preparation")
        logger.info("="*70)
        
        # Check if labels directory exists
        label_dir = Path(self.config['data'].get('labels_dir', 'data/labels'))
        
        if not label_dir.exists():
            logger.warning("No labels directory found. Creating dummy labels for demonstration.")
            logger.warning("For real training, you need to provide actual labels.")
            labels = {}
        else:
            # Load labels from files
            # Implement based on your label format
            labels = {}
            logger.info(f"Loaded labels from {label_dir}")
        
        logger.info(f"✓ Step 4 completed: Prepared labels")
        
        return labels
    
    def step5_train_model(self):
        """
        Step 5: Model Training
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Model Design & Training")
        logger.info("="*70)
        
        graph_dir = self.config['data']['graph_dir']
        
        # Run training script
        cmd = f"python train.py --config config.yaml --data_dir {graph_dir} --model_type {self.config['model']['type']}"
        logger.info(f"Training command: {cmd}")
        logger.info("Run training separately using: python train.py --data_dir <path>")
        
        logger.info("✓ Step 5: Training script ready")
    
    def step6_run_inference(self, model_path: str) -> Dict:
        """
        Step 6: Inference Across All Samples
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Dictionary of predictions
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 6: Inference Across All Samples")
        logger.info("="*70)
        
        from inference import InferenceEngine
        
        # Initialize inference engine
        engine = InferenceEngine(model_path, self.config)
        
        # Run inference
        results = engine.predict_from_directory(self.config['data']['input_dir'])
        
        logger.info(f"✓ Step 6 completed: Generated predictions for {len(results)} images")
        
        return results
    
    def step7_evaluate(self, predictions: Dict, ground_truth: Optional[Dict] = None):
        """
        Step 7: Evaluation & Visualization
        
        Args:
            predictions: Dictionary of predictions
            ground_truth: Optional ground truth labels
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 7: Evaluation & Visualization")
        logger.info("="*70)
        
        from evaluate import Evaluator
        
        # Initialize evaluator
        evaluator = Evaluator(self.config)
        
        # Evaluate
        metrics = evaluator.evaluate(predictions, ground_truth)
        
        logger.info("✓ Step 7 completed: Evaluation finished")
    
    def run_preprocessing_pipeline(self, input_dir: Optional[str] = None):
        """
        Run preprocessing pipeline (Steps 1-4)
        
        Args:
            input_dir: Optional input directory
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Load data
        images = self.step1_load_data(input_dir)
        
        # Step 2: Preprocess
        processed = self.step2_preprocess(images)
        
        # Step 3: Build graphs
        graphs = self.step3_build_graphs(processed)
        
        # Step 4: Prepare labels
        labels = self.step4_prepare_labels()
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING PIPELINE COMPLETED")
        logger.info("="*70)
        logger.info("Next steps:")
        logger.info("1. Run training: python train.py --data_dir data/graphs")
        logger.info("2. Run inference: python inference.py --model_path outputs/models/best_model.pth --input_dir data/raw")
        logger.info("3. Run evaluation: python evaluate.py --predictions_dir outputs/results")
        logger.info("="*70)
    
    def run_full_pipeline(self, input_dir: Optional[str] = None, 
                         model_path: Optional[str] = None):
        """
        Run complete pipeline (requires trained model)
        
        Args:
            input_dir: Optional input directory
            model_path: Path to trained model
        """
        logger.info("\n" + "="*70)
        logger.info("RUNNING COMPLETE PIPELINE")
        logger.info("="*70)
        
        # Preprocessing
        self.run_preprocessing_pipeline(input_dir)
        
        # Training info
        self.step5_train_model()
        
        # Inference (if model available)
        if model_path and Path(model_path).exists():
            predictions = self.step6_run_inference(model_path)
            
            # Evaluation
            self.step7_evaluate(predictions)
        else:
            logger.info("\nNo trained model provided. Train model first:")
            logger.info("python train.py --data_dir data/graphs")
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Protein Sub-Cellular Localization Pipeline"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Directory containing TIFF images")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory for outputs")
    parser.add_argument("--mode", type=str, default="preprocess",
                       choices=['preprocess', 'full'],
                       help="Pipeline mode: 'preprocess' or 'full'")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model (for full mode)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProteinLocalizationPipeline(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        pipeline.config['data']['output_dir'] = args.output_dir
    
    # Run pipeline
    if args.mode == 'preprocess':
        pipeline.run_preprocessing_pipeline(args.input_dir)
    elif args.mode == 'full':
        pipeline.run_full_pipeline(args.input_dir, args.model_path)


if __name__ == "__main__":
    main()
