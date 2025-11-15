"""
Complete pipeline for protein sub-cellular localization
Integrates all components: loading, preprocessing, graph construction, 
training, and visualization
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
import pickle

# Add scripts to path
sys.path.insert(0, os.path.dirname(__file__))

from tiff_loader import TIFFLoader
from preprocessing import ImagePreprocessor
from graph_construction import GraphConstructor
from model_training import GraphCNN, ModelTrainer
from visualization import GraphVisualizer

import torch
from torch_geometric.data import Data, DataLoader


class ProteinLocalizationPipeline:
    """Complete pipeline for protein localization prediction"""
    
    def __init__(self, input_dir: str, output_dir: str, max_files: int = None):
        """
        Initialize pipeline
        
        Args:
            input_dir: Directory containing TIFF images
            output_dir: Directory for outputs
            max_files: Maximum number of files to process (None = all files)
        """
        self.input_dir = input_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        
        # Create subdirectories
        (self.output_dir / 'graphs').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        
        # Initialize components
        self.loader = TIFFLoader(input_dir)
        self.preprocessor = ImagePreprocessor()
        self.graph_constructor = GraphConstructor()
        self.visualizer = GraphVisualizer()
        
        self.graphs = []
        self.features_list = []
        self.synthetic_labels = []  # For synthetic data generation
        
    def load_and_preprocess(self):
        """Load TIFF files and preprocess them"""
        print("=" * 60)
        print("STEP 1: Loading and Preprocessing Images")
        print("=" * 60)
        
        # Check if directory exists
        if not os.path.exists(self.input_dir):
            print(f"Warning: Input directory {self.input_dir} does not exist")
            print("Creating synthetic data for demonstration...")
            return self._create_synthetic_data()
        
        # Scan for TIFF files
        tiff_files = self.loader.scan_directory()
        
        if not tiff_files:
            print("No TIFF files found. Creating synthetic data...")
            return self._create_synthetic_data()
        
        # Process each TIFF file
        all_features = []
        files_to_process = tiff_files if self.max_files is None else tiff_files[:self.max_files]
        total_files = len(files_to_process)
        
        print(f"\nProcessing {total_files} TIFF files from all subdirectories...")
        
        for i, file_path in enumerate(files_to_process):
            print(f"\nProcessing {i+1}/{total_files}: {file_path.name}")
            print(f"  Location: {file_path.parent.name}/")
            
            image = self.loader.load_single_tiff(file_path)
            if image is not None:
                labeled_regions, features = self.preprocessor.process_image(image)
                all_features.append(features)
                
                # Save processed data
                output_path = self.output_dir / 'data' / f'features_{i}.pkl'
                with open(output_path, 'wb') as f:
                    pickle.dump(features, f)
        
        self.features_list = all_features
        print(f"\n✓ Preprocessed {len(all_features)} images from all protein folders")
        return all_features
    
    def _create_synthetic_data(self):
        """Create synthetic data for demonstration with realistic class patterns"""
        print("\nGenerating synthetic data with realistic class patterns...")
        
        all_features = []
        self.synthetic_labels = []  # Store class labels for synthetic data
        np.random.seed(42)
        
        # Define 5 protein localization classes with characteristic features
        class_templates = {
            0: {'name': 'Nucleus', 'area_mean': 800, 'intensity_mean': 0.7, 'eccentricity_mean': 0.3},
            1: {'name': 'Mitochondria', 'area_mean': 200, 'intensity_mean': 0.5, 'eccentricity_mean': 0.7},
            2: {'name': 'ER', 'area_mean': 500, 'intensity_mean': 0.4, 'eccentricity_mean': 0.8},
            3: {'name': 'Golgi', 'area_mean': 300, 'intensity_mean': 0.6, 'eccentricity_mean': 0.5},
            4: {'name': 'Cytoplasm', 'area_mean': 1000, 'intensity_mean': 0.3, 'eccentricity_mean': 0.2}
        }
        
        for i in range(50):  # Increase to 50 samples for better training
            # Assign class based on patterns
            class_idx = i % 5  # Distribute evenly across classes
            template = class_templates[class_idx]
            self.synthetic_labels.append(class_idx)
            
            # Generate regions with features correlated to class
            num_regions = np.random.randint(3, 8)
            features = []
            
            for j in range(num_regions):
                # Add noise to template features
                area = max(50, int(np.random.normal(template['area_mean'], 100)))
                intensity = np.clip(np.random.normal(template['intensity_mean'], 0.15), 0.1, 1.0)
                eccentricity = np.clip(np.random.normal(template['eccentricity_mean'], 0.15), 0, 1.0)
                
                feature = {
                    'label': j + 1,
                    'area': area,
                    'centroid': (np.random.uniform(0, 100), 
                               np.random.uniform(0, 100)),
                    'mean_intensity': intensity,
                    'max_intensity': min(1.0, intensity + np.random.uniform(0.1, 0.2)),
                    'min_intensity': max(0.0, intensity - np.random.uniform(0.1, 0.2)),
                    'eccentricity': eccentricity,
                    'solidity': np.random.uniform(0.7, 1.0),
                    'bbox': (0, 0, 50, 50)
                }
                features.append(feature)
            
            all_features.append(features)
        
        self.features_list = all_features
        print(f"Created {len(all_features)} synthetic samples with feature-class correlations")
        print(f"Class distribution: {[self.synthetic_labels.count(i) for i in range(5)]}")
        return all_features
    
    def construct_graphs(self):
        """Construct graphs from features"""
        print("\n" + "=" * 60)
        print("STEP 2: Constructing Graphs")
        print("=" * 60)
        
        self.graphs = []
        
        for i, features in enumerate(self.features_list):
            G = self.graph_constructor.create_graph_from_regions(features)
            self.graphs.append(G)
            
            # Save graph
            graph_path = self.output_dir / 'graphs' / f'graph_{i}.gml'
            self.graph_constructor.save_graph(G, str(graph_path))
            
            # Print stats
            if i < 3:
                stats = self.graph_constructor.get_graph_statistics(G)
                print(f"\nGraph {i} statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        
        print(f"\nConstructed {len(self.graphs)} graphs")
        return self.graphs
    
    def prepare_training_data(self):
        """Prepare data for training"""
        print("\n" + "=" * 60)
        print("STEP 3: Preparing Training Data")
        print("=" * 60)
        
        # Convert graphs to PyTorch Geometric format
        data_list = []
        
        # Assign synthetic labels for demonstration
        class_names = ['nucleus', 'mitochondria', 'endoplasmic_reticulum', 
                      'golgi', 'cytoplasm']
        
        for i, G in enumerate(self.graphs):
            # Get features
            feature_matrix = self.graph_constructor.get_node_feature_matrix(G)
            
            # Create edge index
            edge_list = list(G.edges())
            if edge_list:
                edge_index = torch.tensor(
                    [[e[0]-1 for e in edge_list] + [e[1]-1 for e in edge_list],
                     [e[1]-1 for e in edge_list] + [e[0]-1 for e in edge_list]],
                    dtype=torch.long
                )
            else:
                edge_index = torch.tensor([[], []], dtype=torch.long)
            
            # Create data object with proper labels
            x = torch.tensor(feature_matrix, dtype=torch.float)
            
            # Use synthetic labels if available, otherwise use cyclic pattern
            if hasattr(self, 'synthetic_labels') and i < len(self.synthetic_labels):
                y = torch.tensor([self.synthetic_labels[i]], dtype=torch.long)
            else:
                # For real data without labels, use features to create pseudo-labels
                # This uses a simple heuristic: large area + high intensity -> nucleus (0)
                # small area + high eccentricity -> mitochondria (1), etc.
                mean_area = feature_matrix[:, 0].mean() if len(feature_matrix) > 0 else 0
                mean_intensity = feature_matrix[:, 1].mean() if len(feature_matrix) > 0 else 0
                mean_eccentricity = feature_matrix[:, 2].mean() if len(feature_matrix) > 0 else 0
                
                # Simple classification rule
                if mean_area > 600:
                    label = 0 if mean_intensity > 0.5 else 4  # Nucleus or Cytoplasm
                elif mean_eccentricity > 0.6:
                    label = 1 if mean_area < 300 else 2  # Mitochondria or ER
                else:
                    label = 3  # Golgi
                
                y = torch.tensor([label], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        print(f"Prepared {len(data_list)} graph samples")
        print(f"Number of classes: {len(class_names)}")
        if hasattr(self, 'synthetic_labels'):
            print(f"Using synthetic labels with feature-class correlations")
        else:
            print(f"Using feature-based pseudo-labels for unsupervised data")
        
        return data_list, class_names
    
    def train_model(self, data_list, class_names, epochs=50):
        """Train the Graph CNN model"""
        print("\n" + "=" * 60)
        print("STEP 4: Training Graph CNN")
        print("=" * 60)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(data_list, test_size=0.2, 
                                                 random_state=42)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=2, shuffle=False)
        
        # Initialize model
        num_features = 4  # area, mean_intensity, eccentricity, solidity
        num_classes = len(class_names)
        
        model = GraphCNN(num_features, num_classes)
        trainer = ModelTrainer(model)
        trainer.setup_training(learning_rate=0.001)
        
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
        print("\nStarting training...")
        
        # Train
        history = trainer.train(train_loader, test_loader, num_epochs=epochs)
        
        # Save model
        model_path = self.output_dir / 'models' / 'graph_cnn.pt'
        trainer.save_model(str(model_path))
        
        print("\nTraining completed!")
        print(f"Final test accuracy: {history['test_accuracy'][-1]:.4f}")
        
        return model, history, train_loader, test_loader
    
    def visualize_results(self, model, history, test_loader, class_names):
        """Visualize results"""
        print("\n" + "=" * 60)
        print("STEP 5: Generating Visualizations")
        print("=" * 60)
        
        # Plot training history
        history_path = self.output_dir / 'visualizations' / 'training_history.png'
        self.visualizer.plot_training_history(history, str(history_path))
        
        # Visualize graphs with predictions
        model.eval()
        
        for i, G in enumerate(self.graphs[:3]):  # Visualize first 3
            # Create data
            feature_matrix = self.graph_constructor.get_node_feature_matrix(G)
            edge_list = list(G.edges())
            
            if edge_list:
                edge_index = torch.tensor(
                    [[e[0]-1 for e in edge_list] + [e[1]-1 for e in edge_list],
                     [e[1]-1 for e in edge_list] + [e[0]-1 for e in edge_list]],
                    dtype=torch.long
                )
            else:
                edge_index = torch.tensor([[], []], dtype=torch.long)
            
            x = torch.tensor(feature_matrix, dtype=torch.float)
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, batch=batch)
            
            # Predict
            with torch.no_grad():
                output = model(data)
                pred_class = output.argmax(dim=1).item()
            
            # Add prediction to graph
            predictions = {node: class_names[pred_class] for node in G.nodes()}
            
            # Visualize
            viz_path = self.output_dir / 'visualizations' / f'graph_{i}_prediction.png'
            self.visualizer.visualize_graph(G, predictions, str(viz_path))
        
        print(f"\nVisualizations saved to {self.output_dir / 'visualizations'}")
    
    def run_complete_pipeline(self, epochs=50):
        """Run the complete pipeline"""
        print("\n" + "=" * 60)
        print("PROTEIN SUB-CELLULAR LOCALIZATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and preprocess
        features = self.load_and_preprocess()
        
        # Step 2: Construct graphs
        graphs = self.construct_graphs()
        
        # Step 3: Prepare training data
        data_list, class_names = self.prepare_training_data()
        
        # Step 4: Train model
        model, history, train_loader, test_loader = self.train_model(
            data_list, class_names, epochs=epochs
        )
        
        # Step 5: Visualize results
        self.visualize_results(model, history, test_loader, class_names)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print(f"  - Graphs: {self.output_dir / 'graphs'}")
        print(f"  - Models: {self.output_dir / 'models'}")
        print(f"  - Visualizations: {self.output_dir / 'visualizations'}")
        
        return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Protein Sub-Cellular Localization Pipeline'
    )
    parser.add_argument('--input', type=str, 
                       default='D:\\5TH_SEM\\CELLULAR\\input',
                       help='Input directory with TIFF files')
    parser.add_argument('--output', type=str,
                       default='D:\\5TH_SEM\\CELLULAR\\output',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (default: all files)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ProteinLocalizationPipeline(args.input, args.output, max_files=args.max_files)
    
    # Run pipeline
    pipeline.run_complete_pipeline(epochs=args.epochs)


if __name__ == "__main__":
    main()
