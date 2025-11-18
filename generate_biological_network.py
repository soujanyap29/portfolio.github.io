"""
Generate clean biological network diagrams from TIFF images or synthetic data
Creates minimal, scientific, bioinformatics-style visualizations
"""

import sys
import os

# Set matplotlib to use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add scripts to path
sys.path.insert(0, os.path.dirname(__file__))

from tiff_loader import TIFFLoader
from preprocessing import ImagePreprocessor
from graph_construction import GraphConstructor
from visualization import GraphVisualizer


def generate_biological_network_from_tiff(tiff_path: str, output_path: str = None):
    """
    Generate biological network diagram from a TIFF image
    
    Args:
        tiff_path: Path to TIFF file
        output_path: Path to save the output diagram (optional)
    """
    print("=" * 60)
    print("Generating Biological Network Diagram from TIFF")
    print("=" * 60)
    
    # Load TIFF
    print(f"\n1. Loading TIFF image: {tiff_path}")
    loader = TIFFLoader(os.path.dirname(tiff_path))
    image = loader.load_single_tiff(tiff_path)
    
    if image is None:
        print("Error: Could not load TIFF image")
        return None
    
    print(f"   Image shape: {image.shape}")
    
    # Preprocess and segment
    print("\n2. Segmenting cellular structures...")
    preprocessor = ImagePreprocessor()
    labeled_regions, features = preprocessor.process_image(image)
    print(f"   Found {len(features)} regions")
    
    # Construct graph
    print("\n3. Building network graph...")
    constructor = GraphConstructor()
    graph = constructor.create_graph_from_regions(features, distance_threshold=50.0)
    
    stats = constructor.get_graph_statistics(graph)
    print(f"   Nodes: {stats['num_nodes']}")
    print(f"   Edges: {stats['num_edges']}")
    
    # Generate biological network visualization
    print("\n4. Creating biological network diagram...")
    visualizer = GraphVisualizer(figsize=(14, 10))
    
    # Determine central node (highest degree)
    degrees = dict(graph.degree())
    central_node = max(degrees, key=degrees.get) if degrees else None
    
    fig = visualizer.visualize_biological_network(
        graph,
        central_node=central_node,
        save_path=output_path,
        title="Protein Localization Network"
    )
    
    print("\n✓ Biological network diagram generated successfully!")
    
    if not output_path:
        plt.show()
    
    return fig, graph


def generate_demo_biological_network(output_path: str = None):
    """
    Generate a demo biological network with synthetic data
    
    Args:
        output_path: Path to save the output diagram (optional)
    """
    print("=" * 60)
    print("Generating Demo Biological Network Diagram")
    print("=" * 60)
    
    # Create synthetic graph with biological network characteristics
    print("\n1. Creating synthetic biological network...")
    
    # Parameters for realistic biological network
    n_nodes = 15
    np.random.seed(42)
    
    # Create scale-free network (common in biological systems)
    G = nx.barabasi_albert_graph(n_nodes, 2, seed=42)
    
    # Add node attributes
    protein_types = ['Receptor', 'Kinase', 'Transcription Factor', 
                    'Enzyme', 'Structural', 'Transport']
    
    for node in G.nodes():
        G.nodes[node]['area'] = np.random.randint(200, 800)
        G.nodes[node]['mean_intensity'] = np.random.uniform(0.4, 0.9)
        G.nodes[node]['eccentricity'] = np.random.uniform(0.3, 0.8)
        G.nodes[node]['solidity'] = np.random.uniform(0.7, 0.95)
        G.nodes[node]['class_label'] = np.random.choice(protein_types)
    
    # Add some random edges to make it more interesting
    for _ in range(5):
        u, v = np.random.choice(list(G.nodes()), 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    
    print(f"   Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate biological network visualization
    print("\n2. Creating biological network diagram...")
    visualizer = GraphVisualizer(figsize=(14, 10))
    
    # Central node is the one with highest degree (hub)
    degrees = dict(G.degree())
    central_node = max(degrees, key=degrees.get)
    
    fig = visualizer.visualize_biological_network(
        G,
        central_node=central_node,
        save_path=output_path,
        title="Biological Network Diagram - Demo"
    )
    
    print("\n✓ Demo biological network diagram generated successfully!")
    print(f"   Central hub node: {central_node} (degree: {degrees[central_node]})")
    
    if not output_path:
        plt.show()
    
    return fig, G


def main():
    """Main function to demonstrate biological network generation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate clean biological network diagrams'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input TIFF file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='biological_network.png',
        help='Path to save output diagram (default: biological_network.png)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Generate demo network with synthetic data'
    )
    
    args = parser.parse_args()
    
    if args.demo or not args.input:
        # Generate demo
        print("\nGenerating demo biological network...\n")
        generate_demo_biological_network(args.output)
    else:
        # Generate from TIFF
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            print("\nGenerating demo instead...\n")
            generate_demo_biological_network(args.output)
        else:
            generate_biological_network_from_tiff(args.input, args.output)
    
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
