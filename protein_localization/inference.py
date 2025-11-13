"""
Inference script for protein localization prediction
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import pickle
import tifffile

from utils.data_loader import TIFFDataLoader
from utils.preprocessor import ImagePreprocessor
from utils.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Inference engine for protein localization prediction"""
    
    def __init__(self, model_path: str, config: Dict, device: str = 'cuda'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            device: Device to use (cuda/cpu)
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize processors
        self.preprocessor = ImagePreprocessor(config.get('preprocessing'))
        self.graph_builder = GraphBuilder(config.get('graph'))
        
        logger.info(f"Inference engine initialized on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine model type from config
        model_type = checkpoint.get('config', {}).get('model', {}).get('type', 'gnn')
        
        if model_type == 'gnn':
            from models.gnn_model import create_gnn_model
            
            # Get input dimension from checkpoint
            model_state = checkpoint['model_state_dict']
            input_dim = model_state['input_proj.weight'].shape[1]
            
            model = create_gnn_model(checkpoint['config'], input_dim)
            model.load_state_dict(model_state)
        
        elif model_type == 'cnn':
            from models.cnn_model import create_cnn_model
            model = create_cnn_model(checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Loaded {model_type} model from {model_path}")
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image"""
        return self.preprocessor.preprocess(image)
    
    def build_graph(self, image: np.ndarray) -> Dict:
        """Build graph from preprocessed image"""
        return self.graph_builder.build_graph(image)
    
    def predict_single(self, image: np.ndarray) -> Dict:
        """
        Predict localization for a single image
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Build graph
        graph_data = self.build_graph(processed)
        
        # Convert to PyTorch Geometric Data
        x = torch.tensor(graph_data['node_features'], dtype=torch.float)
        edge_index = torch.tensor(graph_data['edges'], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        
        # Create batch
        batch = Batch.from_data_list([data]).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
        
        result = {
            'prediction': pred.cpu().item(),
            'probabilities': probs.cpu().numpy()[0],
            'class_name': self.config.get('labels', {}).get('class_names', [])[pred.cpu().item()]
        }
        
        return result
    
    def predict_batch(self, images: Dict[str, np.ndarray],
                     save_results: bool = True) -> Dict[str, Dict]:
        """
        Predict localization for multiple images
        
        Args:
            images: Dictionary mapping filenames to images
            save_results: Whether to save results
            
        Returns:
            Dictionary of predictions
        """
        logger.info(f"Running inference on {len(images)} images...")
        
        results = {}
        
        for filename, image in tqdm(images.items(), desc="Inference"):
            try:
                result = self.predict_single(image)
                result['filename'] = filename
                results[filename] = result
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        if save_results:
            self._save_results(results)
        
        logger.info(f"Completed inference on {len(results)} images")
        
        return results
    
    def _save_results(self, results: Dict[str, Dict]):
        """Save prediction results"""
        output_dir = Path(self.config.get('data', {}).get('output_dir', 'outputs')) / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle
        output_file = output_dir / 'predictions.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved results to {output_file}")
        
        # Save as CSV for easy viewing
        import csv
        csv_file = output_dir / 'predictions.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Prediction', 'Class Name', 'Max Probability'])
            
            for filename, result in results.items():
                writer.writerow([
                    filename,
                    result['prediction'],
                    result['class_name'],
                    f"{result['probabilities'].max():.4f}"
                ])
        
        logger.info(f"Saved results to {csv_file}")
    
    def predict_from_directory(self, input_dir: str) -> Dict[str, Dict]:
        """
        Load images from directory and predict
        
        Args:
            input_dir: Directory containing TIFF images
            
        Returns:
            Dictionary of predictions
        """
        # Load images
        loader = TIFFDataLoader(input_dir)
        images_dict = loader.load_all(validate=True)
        images = {k: v[0] for k, v in images_dict.items()}
        
        # Predict
        results = self.predict_batch(images)
        
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on protein localization")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input TIFF images")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help="Device to use (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output directory if specified
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    # Override device if specified
    device = args.device or config['inference'].get('device', 'cuda')
    
    # Create inference engine
    engine = InferenceEngine(args.model_path, config, device=device)
    
    # Run inference
    results = engine.predict_from_directory(args.input_dir)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Inference Summary")
    logger.info("="*50)
    logger.info(f"Total images processed: {len(results)}")
    
    # Count predictions per class
    from collections import Counter
    pred_counts = Counter([r['prediction'] for r in results.values()])
    class_names = config.get('labels', {}).get('class_names', [])
    
    logger.info("\nPredictions per class:")
    for pred, count in sorted(pred_counts.items()):
        class_name = class_names[pred] if pred < len(class_names) else f"Class {pred}"
        logger.info(f"  {class_name}: {count}")
    
    logger.info("="*50)
    logger.info("Inference completed successfully!")
