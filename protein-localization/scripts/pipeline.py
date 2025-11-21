"""
Pipeline Module
End-to-end automated pipeline for protein localization analysis.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List
import json
import time
from tqdm import tqdm

from tiff_loader import TIFFLoader
from preprocessing import ImagePreprocessor
from graph_construction import GraphConstructor
from model_training import ModelTrainer, GraphDataset
from visualization import Visualizer


class ProteinLocalizationPipeline:
    """
    Complete pipeline for protein sub-cellular localization analysis.
    """
    
    def __init__(self, 
                 input_dir: str = "/mnt/d/5TH_SEM/CELLULAR/input",
                 output_dir: str = "/mnt/d/5TH_SEM/CELLULAR/output",
                 use_gpu: bool = False):
        """
        Initialize pipeline.
        
        Args:
            input_dir: Directory containing TIFF files
            output_dir: Directory for outputs
            use_gpu: Whether to use GPU acceleration
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_gpu = use_gpu
        
        # Create output subdirectories
        self.features_dir = self.output_dir / "features"
        self.graphs_dir = self.output_dir / "graphs"
        self.models_dir = self.output_dir / "models"
        self.figures_dir = self.output_dir / "figures"
        
        for dir_path in [self.features_dir, self.graphs_dir, self.models_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = TIFFLoader(str(self.input_dir), recursive=True)
        self.preprocessor = ImagePreprocessor(use_gpu=use_gpu)
        self.graph_constructor = GraphConstructor()
        self.visualizer = Visualizer(str(self.figures_dir))
        
        self.results = {}
        
        print(f"Pipeline initialized")
        print(f"  Input: {self.input_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  GPU: {use_gpu}")
    
    def load_data(self, max_files: Optional[int] = None) -> List:
        """
        Load TIFF files.
        
        Args:
            max_files: Maximum number of files to load
            
        Returns:
            List of (image, metadata) tuples
        """
        print("\n=== Step 1: Loading TIFF Files ===")
        
        # Scan directory
        self.loader.scan_directory()
        stats = self.loader.get_statistics()
        
        print(f"Found {stats['total_files']} TIFF files")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        
        # Load files
        data = self.loader.load_all(max_files=max_files)
        
        self.results['n_files'] = len(data)
        
        return data
    
    def preprocess_images(self, data: List) -> List:
        """
        Preprocess images (segmentation and feature extraction).
        
        Args:
            data: List of (image, metadata) tuples
            
        Returns:
            List of (image, masks, features, metadata) tuples
        """
        print("\n=== Step 2: Preprocessing Images ===")
        
        processed_data = []
        
        for i, (image, metadata) in enumerate(tqdm(data, desc="Processing images")):
            basename = Path(metadata['filename']).stem
            
            # Segment and extract features
            masks, features, info = self.preprocessor.process_image(
                image,
                output_dir=str(self.features_dir),
                basename=basename
            )
            
            # Save segmentation visualization
            self.visualizer.plot_segmentation_overlay(
                image, masks,
                title=f"Segmentation: {basename}",
                save_name=f"{basename}_segmentation"
            )
            
            processed_data.append((image, masks, features, metadata))
        
        self.results['n_processed'] = len(processed_data)
        
        return processed_data
    
    def build_graphs(self, processed_data: List) -> List:
        """
        Build graphs from processed data.
        
        Args:
            processed_data: List of (image, masks, features, metadata) tuples
            
        Returns:
            List of (graph, metadata) tuples
        """
        print("\n=== Step 3: Building Graphs ===")
        
        graphs = []
        
        for image, masks, features, metadata in tqdm(processed_data, desc="Building graphs"):
            basename = Path(metadata['filename']).stem
            
            # Build spatial graph
            G = self.graph_constructor.build_spatial_graph(features, method='knn')
            
            # Add morphological edges
            self.graph_constructor.add_morphological_edges(G, features)
            
            # Get statistics
            stats = self.graph_constructor.get_graph_statistics(G)
            print(f"\n{basename}: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
            
            # Save graph
            graph_path = self.graphs_dir / f"{basename}_graph.pkl"
            self.graph_constructor.save_graph(G, str(graph_path))
            
            # Visualize graph
            self.visualizer.plot_graph(
                G,
                save_name=f"{basename}_graph"
            )
            
            graphs.append((G, metadata))
        
        self.results['n_graphs'] = len(graphs)
        
        return graphs
    
    def train_model(self, graphs: List, labels: Optional[List[int]] = None,
                   model_type: str = 'gcn', epochs: int = 100) -> Dict:
        """
        Train classification model.
        
        Args:
            graphs: List of graphs
            labels: List of labels (if None, generates dummy labels)
            model_type: Type of model ('gcn', 'cnn', 'hybrid')
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results
        """
        print("\n=== Step 4: Training Model ===")
        
        # Generate dummy labels if not provided
        if labels is None:
            import numpy as np
            labels = np.random.randint(0, 3, len(graphs))
            print("Warning: Using randomly generated labels for demonstration")
        
        # Convert graphs to PyG format
        print("Converting graphs to PyTorch Geometric format...")
        pyg_graphs = []
        for G, _ in graphs:
            pyg_data = self.graph_constructor.convert_to_pyg(G)
            if pyg_data is not None:
                pyg_graphs.append(pyg_data)
        
        if not pyg_graphs:
            print("Error: Could not convert graphs to PyG format")
            return {}
        
        # Split data
        from sklearn.model_selection import train_test_split
        import torch
        
        train_graphs, test_graphs, train_labels, test_labels = train_test_split(
            pyg_graphs, labels, test_size=0.2, random_state=42
        )
        
        train_graphs, val_graphs, train_labels, val_labels = train_test_split(
            train_graphs, train_labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = GraphDataset(train_graphs, train_labels)
        val_dataset = GraphDataset(val_graphs, val_labels)
        test_dataset = GraphDataset(test_graphs, test_labels)
        
        # Create and train model
        trainer = ModelTrainer(model_type=model_type)
        
        input_dim = pyg_graphs[0].x.shape[1] if hasattr(pyg_graphs[0], 'x') else 10
        output_dim = len(set(labels))
        
        trainer.create_model(input_dim, output_dim, hidden_dim=64, num_layers=2)
        
        # Train
        trainer.train(train_dataset, val_dataset, epochs=epochs, lr=0.001, batch_size=8)
        
        # Evaluate
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        metrics = trainer.compute_metrics(test_loader)
        
        print("\n=== Model Performance ===")
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                print(f"{key}: {value:.4f}")
        
        # Save model
        model_path = self.models_dir / f"{model_type}_model.pt"
        trainer.save_checkpoint(str(model_path))
        
        # Visualize results
        self.visualizer.plot_training_history(trainer.history, save_name=f"{model_type}_training")
        self.visualizer.plot_metrics_summary(metrics, save_name=f"{model_type}_metrics")
        
        # Save metrics
        metrics_path = self.models_dir / f"{model_type}_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = {k: v.tolist() if hasattr(v, 'tolist') else v 
                          for k, v in metrics.items()}
            json.dump(metrics_json, f, indent=2)
        
        self.results['model_type'] = model_type
        self.results['metrics'] = metrics
        
        return metrics
    
    def run(self, max_files: Optional[int] = None, 
            train_model: bool = True,
            model_type: str = 'gcn',
            epochs: int = 50):
        """
        Run complete pipeline.
        
        Args:
            max_files: Maximum number of files to process
            train_model: Whether to train model
            model_type: Type of model to train
            epochs: Number of training epochs
        """
        start_time = time.time()
        
        print("=" * 60)
        print("PROTEIN SUB-CELLULAR LOCALIZATION PIPELINE")
        print("=" * 60)
        
        try:
            # Load data
            data = self.load_data(max_files=max_files)
            
            if not data:
                print("Error: No data loaded")
                return
            
            # Preprocess
            processed_data = self.preprocess_images(data)
            
            # Build graphs
            graphs = self.build_graphs(processed_data)
            
            # Train model
            if train_model and graphs:
                metrics = self.train_model(graphs, model_type=model_type, epochs=epochs)
            
            # Save pipeline results
            elapsed_time = time.time() - start_time
            self.results['elapsed_time'] = elapsed_time
            
            results_path = self.output_dir / "pipeline_results.json"
            with open(results_path, 'w') as f:
                # Convert results for JSON serialization
                results_json = {}
                for k, v in self.results.items():
                    if isinstance(v, dict):
                        results_json[k] = {kk: vv.tolist() if hasattr(vv, 'tolist') else vv 
                                         for kk, vv in v.items()}
                    else:
                        results_json[k] = v
                json.dump(results_json, f, indent=2)
            
            print("\n" + "=" * 60)
            print(f"PIPELINE COMPLETE in {elapsed_time:.2f} seconds")
            print(f"Results saved to: {self.output_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\nError in pipeline: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Protein Localization Pipeline")
    parser.add_argument('--input', type=str, default="/mnt/d/5TH_SEM/CELLULAR/input",
                       help='Input directory with TIFF files')
    parser.add_argument('--output', type=str, default="/mnt/d/5TH_SEM/CELLULAR/output",
                       help='Output directory')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'cnn', 'hybrid'],
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip model training')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ProteinLocalizationPipeline(
        input_dir=args.input,
        output_dir=args.output,
        use_gpu=args.gpu
    )
    
    pipeline.run(
        max_files=args.max_files,
        train_model=not args.no_train,
        model_type=args.model,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
