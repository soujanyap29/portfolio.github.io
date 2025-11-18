#!/usr/bin/env python3
"""
Command-line interface for Protein Localization System

Usage:
    python cli.py segment <tiff_file>           # Segment a single TIFF
    python cli.py graph <tiff_file>             # Build graph from TIFF
    python cli.py predict <tiff_file> <model>   # Predict using trained model
    python cli.py visualize <graph_file>        # Visualize a saved graph
    python cli.py batch <directory>             # Process all TIFFs in directory
"""

import sys
import argparse
from pathlib import Path
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import TIFFProcessor
from graph_construction import BiologicalGraphBuilder
from visualization import ScientificVisualizer
from models import GraphCNN, ModelTrainer
from utils import load_config, ensure_dir, get_tiff_files
import torch


def segment_command(args):
    """Segment a TIFF file"""
    config = load_config(args.config)
    processor = TIFFProcessor(config)
    
    print(f"Segmenting: {args.tiff_file}")
    img, masks, features = processor.process_single_tiff(args.tiff_file)
    
    # Save results
    output_path = args.output or f"{Path(args.tiff_file).stem}_segmented.png"
    
    visualizer = ScientificVisualizer(config)
    visualizer.plot_segmentation_overlay(img, masks, output_path)
    
    print(f"✓ Segmentation saved to: {output_path}")
    print(f"✓ Found {len(features['region_ids'])} regions")


def graph_command(args):
    """Build graph from TIFF file"""
    config = load_config(args.config)
    processor = TIFFProcessor(config)
    graph_builder = BiologicalGraphBuilder(config)
    
    print(f"Building graph from: {args.tiff_file}")
    img, masks, features = processor.process_single_tiff(args.tiff_file)
    G = graph_builder.build_graph(features)
    
    # Save graph
    output_path = args.output or f"{Path(args.tiff_file).stem}_graph.gpickle"
    graph_builder.save_graph(G, output_path)
    
    print(f"✓ Graph saved to: {output_path}")
    print(f"✓ Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Optionally visualize
    if args.visualize:
        viz_path = f"{Path(output_path).stem}.png"
        visualizer = ScientificVisualizer(config)
        visualizer.plot_graph_visualization(G, viz_path)
        print(f"✓ Visualization saved to: {viz_path}")


def predict_command(args):
    """Make prediction using trained model"""
    config = load_config(args.config)
    processor = TIFFProcessor(config)
    graph_builder = BiologicalGraphBuilder(config)
    
    print(f"Processing: {args.tiff_file}")
    img, masks, features = processor.process_single_tiff(args.tiff_file)
    G = graph_builder.build_graph(features)
    
    # Convert to PyTorch Geometric format
    dummy_labels = [0] * G.number_of_nodes()
    graph_data = graph_builder.graph_to_pytorch_geometric(G, dummy_labels)
    
    # Load model
    print(f"Loading model: {args.model_file}")
    num_features = graph_data.x.shape[1]
    num_classes = args.num_classes
    
    model = GraphCNN(num_features, num_classes)
    checkpoint = torch.load(args.model_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Predict
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        pred_class = out.argmax(dim=1).item()
        probabilities = torch.softmax(out, dim=1).squeeze().tolist()
    
    # Output results
    class_names = args.class_names.split(',') if args.class_names else [f"Class {i}" for i in range(num_classes)]
    
    print(f"\n✨ Prediction Results:")
    print(f"  Predicted Class: {class_names[pred_class]}")
    print(f"  Confidence: {probabilities[pred_class]:.2%}")
    print(f"\n  Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"    {class_names[i]}: {prob:.2%}")
    
    # Save to JSON if requested
    if args.output:
        result = {
            'file': str(args.tiff_file),
            'predicted_class': class_names[pred_class],
            'predicted_index': int(pred_class),
            'confidence': float(probabilities[pred_class]),
            'probabilities': {class_names[i]: float(p) for i, p in enumerate(probabilities)},
            'num_regions': len(features['region_ids']),
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        }
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


def visualize_command(args):
    """Visualize a saved graph"""
    config = load_config(args.config)
    graph_builder = BiologicalGraphBuilder(config)
    visualizer = ScientificVisualizer(config)
    
    print(f"Loading graph: {args.graph_file}")
    G = graph_builder.load_graph(args.graph_file)
    
    output_path = args.output or f"{Path(args.graph_file).stem}_visualization.png"
    visualizer.plot_graph_visualization(G, output_path, node_labels=args.labels)
    
    print(f"✓ Visualization saved to: {output_path}")


def batch_command(args):
    """Process all TIFFs in directory"""
    config = load_config(args.config)
    processor = TIFFProcessor(config)
    graph_builder = BiologicalGraphBuilder(config)
    visualizer = ScientificVisualizer(config)
    
    tiff_files = get_tiff_files(args.directory, recursive=args.recursive)
    print(f"Found {len(tiff_files)} TIFF files")
    
    output_dir = args.output or './batch_output'
    ensure_dir(output_dir)
    
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}] Processing: {Path(tiff_file).name}")
        
        try:
            # Process
            img, masks, features = processor.process_single_tiff(tiff_file)
            G = graph_builder.build_graph(features)
            
            # Save outputs
            base_name = Path(tiff_file).stem
            
            if args.save_segmentation:
                seg_path = f"{output_dir}/{base_name}_segmentation.png"
                visualizer.plot_segmentation_overlay(img, masks, seg_path)
            
            if args.save_graph:
                graph_path = f"{output_dir}/{base_name}_graph.gpickle"
                graph_builder.save_graph(G, graph_path)
                
                if args.save_visualization:
                    viz_path = f"{output_dir}/{base_name}_graph_viz.png"
                    visualizer.plot_graph_visualization(G, viz_path)
            
            print(f"  ✓ Regions: {len(features['region_ids'])}, Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Batch processing complete. Output in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Protein Localization System CLI')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment a TIFF file')
    segment_parser.add_argument('tiff_file', help='Path to TIFF file')
    segment_parser.add_argument('-o', '--output', help='Output path')
    
    # Graph command
    graph_parser = subparsers.add_parser('graph', help='Build graph from TIFF')
    graph_parser.add_argument('tiff_file', help='Path to TIFF file')
    graph_parser.add_argument('-o', '--output', help='Output path')
    graph_parser.add_argument('-v', '--visualize', action='store_true', help='Create visualization')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict using model')
    predict_parser.add_argument('tiff_file', help='Path to TIFF file')
    predict_parser.add_argument('model_file', help='Path to trained model')
    predict_parser.add_argument('-n', '--num-classes', type=int, default=3, help='Number of classes')
    predict_parser.add_argument('-c', '--class-names', help='Comma-separated class names')
    predict_parser.add_argument('-o', '--output', help='Save results to JSON')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize saved graph')
    viz_parser.add_argument('graph_file', help='Path to graph file')
    viz_parser.add_argument('-o', '--output', help='Output path')
    viz_parser.add_argument('-l', '--labels', action='store_true', help='Show node labels')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process directory of TIFFs')
    batch_parser.add_argument('directory', help='Directory containing TIFFs')
    batch_parser.add_argument('-o', '--output', help='Output directory')
    batch_parser.add_argument('-r', '--recursive', action='store_true', help='Scan recursively')
    batch_parser.add_argument('--save-segmentation', action='store_true', help='Save segmentation')
    batch_parser.add_argument('--save-graph', action='store_true', help='Save graphs')
    batch_parser.add_argument('--save-visualization', action='store_true', help='Save visualizations')
    
    args = parser.parse_args()
    
    if args.command == 'segment':
        segment_command(args)
    elif args.command == 'graph':
        graph_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    elif args.command == 'batch':
        batch_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
