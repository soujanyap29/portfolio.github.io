#!/usr/bin/env python
"""
Quick demo script to test the protein localization pipeline with synthetic data.
Run this to verify installation and see the pipeline in action.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import TIFFPreprocessor
from graph_builder import BiologicalGraphBuilder
from models import ModelTrainer
from visualization import ProteinVisualization


def create_synthetic_data(output_dir):
    """Create synthetic test data."""
    print("Creating synthetic test data...")
    
    synthetic_results = []
    
    for i in range(3):
        # Create synthetic features
        features = []
        for label in range(1, 8):
            features.append({
                'label': label,
                'centroid_y': np.random.rand() * 256,
                'centroid_x': np.random.rand() * 256,
                'distance_from_center': np.random.rand() * 100,
                'area': np.random.rand() * 500 + 100,
                'perimeter': np.random.rand() * 100 + 50,
                'eccentricity': np.random.rand(),
                'solidity': np.random.rand() * 0.3 + 0.7,
                'extent': np.random.rand(),
                'major_axis_length': np.random.rand() * 50,
                'minor_axis_length': np.random.rand() * 30,
                'orientation': np.random.rand() * np.pi,
                'mean_intensity': np.random.rand() * 200 + 50,
                'max_intensity': np.random.rand() * 255,
                'min_intensity': np.random.rand() * 50,
                'intensity_std': np.random.rand() * 30,
                'bbox_min_row': 0, 'bbox_min_col': 0,
                'bbox_max_row': 256, 'bbox_max_col': 256
            })
        
        synthetic_results.append({
            'file_path': f'demo_sample_{i}.tif',
            'file_name': f'demo_sample_{i}.tif',
            'image_shape': (256, 256),
            'masks': np.random.randint(0, 8, (256, 256)),
            'features': features,
            'segmentation_metadata': {'n_cells': len(features), 'method': 'synthetic'},
            'n_regions': len(features)
        })
    
    return synthetic_results


def main():
    """Run demo pipeline."""
    print("=" * 70)
    print("PROTEIN LOCALIZATION PIPELINE - DEMO")
    print("=" * 70)
    
    # Setup
    output_dir = Path(__file__).parent / "demo_output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Create synthetic data
    print("\n" + "=" * 70)
    print("STEP 1: Data Preparation")
    print("=" * 70)
    processed_results = create_synthetic_data(output_dir)
    print(f"✓ Created {len(processed_results)} synthetic samples")
    
    # Step 2: Build graphs
    print("\n" + "=" * 70)
    print("STEP 2: Graph Construction")
    print("=" * 70)
    graph_builder = BiologicalGraphBuilder(distance_threshold=50.0, k_neighbors=3)
    graph_results = graph_builder.process_results(processed_results, str(output_dir / "graphs"))
    print(f"✓ Built {len(graph_results)} biological graphs")
    
    # Display graph stats
    for i, result in enumerate(graph_results):
        print(f"  Graph {i+1}: {result['n_nodes']} nodes, {result['n_edges']} edges")
    
    # Step 3: Train model
    print("\n" + "=" * 70)
    print("STEP 3: Model Training")
    print("=" * 70)
    
    pyg_data_list = [result['pyg_data'] for result in graph_results]
    
    # Add dummy labels for demo
    import torch
    for data in pyg_data_list:
        data.y = torch.randint(0, 3, (1,))
    
    # Quick training (just 5 epochs for demo)
    from torch_geometric.data import DataLoader
    train_loader = DataLoader(pyg_data_list, batch_size=2, shuffle=True)
    
    trainer = ModelTrainer(model_type='graph_cnn', num_classes=3)
    trainer.create_model(num_node_features=pyg_data_list[0].x.shape[1])
    
    print("Training model (5 epochs for demo)...")
    trainer.train(train_loader, train_loader, epochs=5, lr=0.01)
    
    # Save model
    model_path = output_dir / "models"
    model_path.mkdir(exist_ok=True)
    trainer.save_model(str(model_path / "demo_model.pt"))
    
    # Step 4: Create visualizations
    print("\n" + "=" * 70)
    print("STEP 4: Visualization")
    print("=" * 70)
    
    viz = ProteinVisualization(str(output_dir / "visualizations"))
    
    for i, graph_result in enumerate(graph_results):
        print(f"Creating visualization for sample {i+1}...")
        G = graph_result['networkx_graph']
        viz.plot_graph(G, filename=f"demo_graph_{i}.png", show_labels=True)
        
        # Plot features
        viz.plot_feature_distributions(
            processed_results[i]['features'],
            filename=f"demo_features_{i}.png"
        )
    
    # Plot training history
    viz.plot_training_history(trainer.history, filename="demo_training.png")
    
    print("✓ Visualizations created")
    
    # Step 5: Make predictions
    print("\n" + "=" * 70)
    print("STEP 5: Prediction Demo")
    print("=" * 70)
    
    sample = pyg_data_list[0]
    trainer.model.eval()
    
    with torch.no_grad():
        from torch_geometric.data import Batch
        batch_sample = Batch.from_data_list([sample])
        output = trainer.model(batch_sample)
        pred = output.argmax(dim=1).item()
        probs = torch.exp(output)[0]
        confidence = probs[pred].item()
    
    print(f"\nSample: {processed_results[0]['file_name']}")
    print(f"Predicted Class: {pred}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nClass Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  Class {i}: {prob.item() * 100:.2f}%")
    
    # Final summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Models: {model_path}")
    print(f"  - Graphs: {output_dir / 'graphs'}")
    print(f"  - Visualizations: {output_dir / 'visualizations'}")
    print("\n✓ Pipeline demo completed successfully!")
    print("\nNext steps:")
    print("  1. Run the Jupyter notebook: jupyter lab notebooks/final_pipeline.ipynb")
    print("  2. Launch the web interface: cd frontend && python app.py")
    print("  3. Add your own TIFF files to the input directory")


if __name__ == "__main__":
    main()
