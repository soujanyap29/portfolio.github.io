#!/usr/bin/env python3
"""
Complete pipeline example script

This script demonstrates the complete workflow from TIFF loading
to model prediction and visualization.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from preprocessing import TIFFProcessor
from graph_construction import BiologicalGraphBuilder
from models import GraphCNN, ModelTrainer
from visualization import ScientificVisualizer
from utils import load_config, ensure_dir, get_tiff_files
from utils.metrics import MetricsCalculator, calculate_colocalization_metrics

import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split


def main():
    """Run complete pipeline"""
    
    print("="*70)
    print("PROTEIN SUB-CELLULAR LOCALIZATION PIPELINE")
    print("="*70)
    
    # 1. Load configuration
    print("\n[1/8] Loading configuration...")
    config = load_config('config.yaml')
    
    input_dir = config['data']['input_dir']
    output_dir = config['data']['output_dir']
    
    # Create output directories
    for subdir in ['models', 'visualizations', 'segmented', 'graphs', 'predictions']:
        ensure_dir(os.path.join(output_dir, subdir))
    
    print(f"  ✓ Input: {input_dir}")
    print(f"  ✓ Output: {output_dir}")
    
    # 2. Find TIFF files
    print("\n[2/8] Scanning for TIFF files...")
    
    # Create sample data if input directory doesn't exist
    if not os.path.exists(input_dir):
        print(f"  ! Input directory not found, creating sample data...")
        import tifffile
        sample_dir = './sample_data'
        ensure_dir(sample_dir)
        
        for i in range(5):
            sample = np.random.randint(0, 255, (5, 10, 256, 256), dtype=np.uint8)
            tifffile.imwrite(f'{sample_dir}/sample_{i}.tif', sample)
        
        input_dir = sample_dir
        config['data']['input_dir'] = sample_dir
    
    tiff_files = get_tiff_files(input_dir, recursive=True)
    print(f"  ✓ Found {len(tiff_files)} TIFF files")
    
    if len(tiff_files) == 0:
        print("  ! No TIFF files found. Exiting.")
        return
    
    # Limit to first 5 for demo
    tiff_files = tiff_files[:5]
    
    # 3. Preprocessing and segmentation
    print("\n[3/8] Processing and segmenting TIFF files...")
    processor = TIFFProcessor(config)
    
    all_images = []
    all_masks = []
    all_features = []
    
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"  Processing {i}/{len(tiff_files)}: {Path(tiff_file).name}")
        try:
            img, masks, features = processor.process_single_tiff(tiff_file)
            all_images.append(img)
            all_masks.append(masks)
            all_features.append(features)
        except Exception as e:
            print(f"  ! Error: {e}")
    
    print(f"  ✓ Processed {len(all_features)} files")
    print(f"  ✓ Total regions: {sum(len(f['region_ids']) for f in all_features)}")
    
    # 4. Graph construction
    print("\n[4/8] Building biological graphs...")
    graph_builder = BiologicalGraphBuilder(config)
    
    all_graphs = []
    all_graph_data = []
    
    for i, features in enumerate(all_features):
        G = graph_builder.build_graph(features)
        all_graphs.append(G)
        
        # Generate random labels for demo (replace with actual labels)
        num_nodes = G.number_of_nodes()
        labels = [np.random.randint(0, 3) for _ in range(num_nodes)]
        
        graph_data = graph_builder.graph_to_pytorch_geometric(G, labels)
        if graph_data is not None:
            all_graph_data.append(graph_data)
        
        # Save graph
        graph_path = os.path.join(output_dir, 'graphs', f'graph_{i}.gpickle')
        graph_builder.save_graph(G, graph_path)
    
    print(f"  ✓ Built {len(all_graphs)} graphs")
    print(f"  ✓ Avg nodes: {np.mean([G.number_of_nodes() for G in all_graphs]):.1f}")
    print(f"  ✓ Avg edges: {np.mean([G.number_of_edges() for G in all_graphs]):.1f}")
    
    # 5. Model training
    print("\n[5/8] Training model...")
    
    if len(all_graph_data) > 2:
        # Split data
        train_data, test_data = train_test_split(
            all_graph_data, 
            test_size=0.2,
            random_state=42
        )
        
        train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=2)
        
        # Initialize model
        num_features = all_graph_data[0].x.shape[1]
        num_classes = 3
        
        model = GraphCNN(num_features, num_classes, hidden_dim=32)
        trainer = ModelTrainer(model, config)
        
        print(f"  ✓ Model: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train for a few epochs (demo)
        print("  Training for 5 epochs...")
        for epoch in range(5):
            loss, acc = trainer.train_epoch(train_loader)
            if (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.4f}")
        
        # Save model
        model_path = os.path.join(output_dir, 'models', 'graph_cnn_demo.pth')
        trainer.save_model(model_path)
        print(f"  ✓ Model saved to {model_path}")
        
        # 6. Evaluation
        print("\n[6/8] Evaluating model...")
        test_acc, predictions, labels = trainer.evaluate(test_loader)
        
        # Calculate comprehensive metrics
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all_metrics(
            labels, 
            predictions,
            class_names=['Mitochondrial', 'Nuclear', 'Cytoplasmic']
        )
        
        metrics_calc.print_metrics()
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'predictions', 'metrics.json')
        metrics_calc.save_metrics(metrics_path)
        
    else:
        print("  ! Not enough data for training. Skipping model training.")
        model = None
        predictions = None
        labels = None
    
    # 7. Visualization
    print("\n[7/8] Generating visualizations...")
    visualizer = ScientificVisualizer(config)
    viz_dir = os.path.join(output_dir, 'visualizations')
    
    if len(all_images) > 0:
        # Segmentation overlay
        print("  Creating segmentation overlay...")
        visualizer.plot_segmentation_overlay(
            all_images[0],
            all_masks[0],
            os.path.join(viz_dir, 'segmentation_overlay.png')
        )
        
        # Compartment map
        print("  Creating compartment map...")
        visualizer.plot_compartment_map(
            all_masks[0],
            os.path.join(viz_dir, 'compartment_map.png')
        )
    
    if len(all_graphs) > 0:
        # Graph visualization
        print("  Creating graph visualization...")
        visualizer.plot_graph_visualization(
            all_graphs[0],
            os.path.join(viz_dir, 'graph_visualization.png'),
            node_labels=True
        )
    
    if predictions is not None and labels is not None:
        # Confusion matrix
        print("  Creating confusion matrix...")
        visualizer.plot_confusion_matrix(
            labels,
            predictions,
            ['Mitochondrial', 'Nuclear', 'Cytoplasmic'],
            os.path.join(viz_dir, 'confusion_matrix.png')
        )
    
    # Statistical plots
    if len(all_features) > 0:
        print("  Creating statistical plots...")
        
        # Grouped bar plot
        grouped_data = {
            'Low': [f for feat in all_features for f in feat['mean_intensities'] if f < 0.3],
            'Medium': [f for feat in all_features for f in feat['mean_intensities'] if 0.3 <= f < 0.7],
            'High': [f for feat in all_features for f in feat['mean_intensities'] if f >= 0.7]
        }
        
        if all(grouped_data.values()):
            visualizer.plot_grouped_bar(
                grouped_data,
                os.path.join(viz_dir, 'intensity_grouped_bar.png'),
                ylabel='Mean Intensity'
            )
            
            visualizer.plot_box_violin(
                grouped_data,
                os.path.join(viz_dir, 'intensity_box_violin.png'),
                ylabel='Mean Intensity'
            )
    
    print("  ✓ Visualizations complete")
    
    # 8. Prediction demo
    print("\n[8/8] Running prediction demo...")
    
    if len(tiff_files) > 0 and model is not None:
        sample_file = tiff_files[0]
        print(f"  Processing: {Path(sample_file).name}")
        
        img, masks, features = processor.process_single_tiff(sample_file)
        G = graph_builder.build_graph(features)
        
        dummy_labels = [0] * G.number_of_nodes()
        graph_data = graph_builder.graph_to_pytorch_geometric(G, dummy_labels)
        
        if graph_data is not None:
            model.eval()
            with torch.no_grad():
                graph_data = graph_data.to(trainer.device)
                out = model(graph_data.x, graph_data.edge_index)
                pred_class = out.argmax(dim=1).item()
                confidence = torch.softmax(out, dim=1).max().item()
            
            class_names = ['Mitochondrial', 'Nuclear', 'Cytoplasmic']
            print(f"\n  Prediction Results:")
            print(f"    Class: {class_names[pred_class]}")
            print(f"    Confidence: {confidence:.2%}")
            print(f"    Regions: {len(features['region_ids'])}")
            print(f"    Avg Area: {np.mean(features['areas']):.1f} pixels²")
            print(f"    Graph Nodes: {G.number_of_nodes()}")
            print(f"    Graph Edges: {G.number_of_edges()}")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Models: {output_dir}/models/")
    print(f"  - Visualizations: {output_dir}/visualizations/")
    print(f"  - Graphs: {output_dir}/graphs/")
    print(f"  - Predictions: {output_dir}/predictions/")
    print("\nTo view visualizations, check the output directory.")
    print("To start the web interface, run: python frontend/app.py")
    print("="*70)


if __name__ == '__main__':
    main()
