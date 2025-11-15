#!/usr/bin/env python3
"""
Main execution script for Protein Sub-Cellular Localization Pipeline
"""
import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.segmentation import DirectoryHandler, TIFFLoader, CellposeSegmenter
from preprocessing.feature_extraction import FeatureExtractor, FeatureStorage
from graph_construction.graph_builder import GraphConstructor, GraphStorage
from visualization.plotters import SegmentationVisualizer
from visualization.graph_viz import GraphVisualizer
import config


def process_single_file(input_file: str, output_dir: str):
    """Process a single TIFF file"""
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}\n")
    
    # Load
    print("1. Loading TIFF file...")
    loader = TIFFLoader()
    image = loader.load_tiff(input_file)
    if image is None:
        print("❌ Error loading file")
        return
    print(f"✓ Loaded image shape: {image.shape}")
    
    # Segment
    print("\n2. Segmenting image...")
    segmenter = CellposeSegmenter(model_type=config.CELLPOSE_MODEL)
    masks, seg_info = segmenter.segment_image(image)
    if masks is None:
        print("❌ Segmentation failed")
        return
    print(f"✓ Detected {seg_info['num_cells']} cells/regions")
    
    # Extract features
    print("\n3. Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(image, masks)
    if features.empty:
        print("❌ Feature extraction failed")
        return
    print(f"✓ Extracted {len(features.columns)} features from {len(features)} regions")
    
    # Build graph
    print("\n4. Constructing graph...")
    constructor = GraphConstructor(
        proximity_threshold=config.PROXIMITY_THRESHOLD,
        max_edges_per_node=config.MAX_EDGES_PER_NODE
    )
    graph = constructor.construct_graph(features, masks)
    print(f"✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Save results
    print("\n5. Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    feature_storage = FeatureStorage(output_dir)
    filename = Path(input_file).stem
    feature_storage.save_features(features, f"{filename}_features")
    
    # Save graph
    graph_storage = GraphStorage(output_dir)
    graph_storage.save_graph(graph, filename)
    
    # Create visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    seg_viz = SegmentationVisualizer(output_dir=viz_dir)
    seg_viz.plot_segmentation_overlay(image, masks, 
                                     title=f"Segmentation: {filename}",
                                     filename=f"{filename}_segmentation.png")
    
    graph_viz = GraphVisualizer(output_dir=viz_dir)
    graph_viz.plot_graph(graph, 
                        title=f"Graph: {filename}",
                        filename=f"{filename}_graph.png")
    
    print(f"\n✅ Processing complete!")
    print(f"   Features: {output_dir}/{filename}_features.csv")
    print(f"   Graph: {output_dir}/{filename}.gpickle")
    print(f"   Visualizations: {viz_dir}/")


def process_directory(input_dir: str, output_dir: str, max_files: int = None):
    """Process all TIFF files in directory"""
    print(f"\n{'='*60}")
    print(f"Processing directory: {input_dir}")
    print(f"{'='*60}\n")
    
    # Scan for files
    print("Scanning for TIFF files...")
    handler = DirectoryHandler(input_dir, config.TIFF_EXTENSIONS)
    tiff_files = handler.scan_directory()
    
    if not tiff_files:
        print("❌ No TIFF files found")
        return
    
    if max_files:
        tiff_files = tiff_files[:max_files]
    
    print(f"Found {len(tiff_files)} files to process\n")
    
    # Process each file
    for i, tiff_file in enumerate(tiff_files, 1):
        print(f"\n[{i}/{len(tiff_files)}] Processing file...")
        try:
            process_single_file(tiff_file, output_dir)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ Completed processing {len(tiff_files)} files")
    print(f"{'='*60}")


def launch_interface(model_path: str = None, output_dir: str = None):
    """Launch web interface"""
    from interface.app import launch_interface as launch
    output = output_dir or config.OUTPUT_DIR
    print("\n🚀 Launching web interface...")
    print(f"   Server: http://localhost:7860")
    print(f"   Output directory: {output}")
    launch(model_path=model_path, output_dir=output, share=False)


def main():
    parser = argparse.ArgumentParser(
        description="Protein Sub-Cellular Localization Pipeline"
    )
    
    parser.add_argument(
        'command',
        choices=['process', 'interface', 'notebook'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input file or directory path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=config.OUTPUT_DIR,
        help='Output directory'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model'
    )
    
    args = parser.parse_args()
    
    if args.command == 'process':
        if not args.input:
            print("❌ Error: --input required for process command")
            sys.exit(1)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            process_single_file(str(input_path), args.output)
        elif input_path.is_dir():
            process_directory(str(input_path), args.output, args.max_files)
        else:
            print(f"❌ Error: {input_path} not found")
            sys.exit(1)
    
    elif args.command == 'interface':
        launch_interface(model_path=args.model, output_dir=args.output)
    
    elif args.command == 'notebook':
        print("\n📓 Starting Jupyter Lab...")
        print("   Opening notebooks/final_pipeline.ipynb")
        os.system("jupyter lab notebooks/final_pipeline.ipynb")


if __name__ == "__main__":
    main()
