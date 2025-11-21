"""
Streamlit Web Interface for Protein Localization Pipeline
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import json
import os

# Add scripts directory to path - try multiple approaches for robustness
try:
    # Method 1: Resolve from __file__
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if not scripts_dir.exists():
        raise FileNotFoundError(f"Scripts directory not found at {scripts_dir}")
except:
    # Method 2: Resolve from current working directory
    scripts_dir = Path.cwd() / "scripts"
    if not scripts_dir.exists():
        # Method 3: Go up from current directory
        scripts_dir = Path.cwd().parent / "scripts"
        if not scripts_dir.exists():
            st.error(f"Cannot find scripts directory. Please ensure you're running from the protein-localization directory.")
            st.stop()

# Add to path if not already there
scripts_dir_str = str(scripts_dir)
if scripts_dir_str not in sys.path:
    sys.path.insert(0, scripts_dir_str)

# Try to import modules with better error handling
try:
    from tiff_loader import load_tiff_from_path
    from preprocessing import ImagePreprocessor
    from graph_construction import GraphConstructor
    from visualization import Visualizer
except ImportError as e:
    st.error(f"""
    **Import Error**: {str(e)}
    
    Please ensure:
    1. You are running the app from the correct directory
    2. All dependencies are installed: `pip install -r requirements.txt`
    3. The scripts directory exists at: `{scripts_dir}`
    
    **To fix:**
    ```bash
    cd protein-localization
    pip install -r requirements.txt
    streamlit run frontend/streamlit_app.py
    ```
    """)
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Protein Localization Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🔬 Protein Sub-Cellular Localization Pipeline</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Upload TIFF microscopy images to analyze protein sub-cellular localization using 
    deep learning and graph neural networks.
    """)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Protein+Pipeline", 
                use_column_width=True)
        
        st.markdown("### Pipeline Settings")
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=False)
        
        st.markdown("### Segmentation")
        cell_diameter = st.slider("Expected Cell Diameter (pixels)", 10, 100, 30)
        
        st.markdown("### Graph Construction")
        distance_threshold = st.slider("Distance Threshold", 10, 200, 50)
        k_neighbors = st.slider("K Neighbors", 2, 10, 5)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This pipeline performs:
        - TIFF image loading
        - Cell segmentation (Cellpose)
        - Feature extraction
        - Graph construction
        - GNN-based classification
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Results", "📈 Visualizations", "ℹ️ Help"])
    
    with tab1:
        upload_and_process_tab(use_gpu, cell_diameter, distance_threshold, k_neighbors)
    
    with tab2:
        results_tab()
    
    with tab3:
        visualizations_tab()
    
    with tab4:
        help_tab()


def upload_and_process_tab(use_gpu, cell_diameter, distance_threshold, k_neighbors):
    """Upload and processing tab."""
    
    st.markdown('<div class="section-header">Upload TIFF File</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a TIFF file",
        type=['tif', 'tiff'],
        help="Upload a microscopy TIFF image (supports multi-channel and 3D/4D)"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("Type", uploaded_file.type)
        
        # Process button
        if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
            run_pipeline(uploaded_file, use_gpu, cell_diameter, distance_threshold, k_neighbors)


def run_pipeline(uploaded_file, use_gpu, cell_diameter, distance_threshold, k_neighbors):
    """Run the complete pipeline on uploaded file."""
    
    tmp_path = None
    
    try:
        with st.spinner("Processing image..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())  # Use getvalue() instead of read()
                tmp_path = tmp_file.name
            
            # Create progress tracker
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load TIFF
            status_text.text("Loading TIFF file...")
            progress_bar.progress(10)
            
            image, metadata = load_tiff_from_path(tmp_path)
            
            if image is None:
                st.error("Failed to load TIFF file")
                return
            
            st.success(f"Loaded image: {metadata['shape']}, {metadata['dtype']}")
            
            # Step 2: Preprocess
            status_text.text("Segmenting image...")
            progress_bar.progress(30)
            
            preprocessor = ImagePreprocessor(use_gpu=use_gpu)
            preprocessor.load_cellpose_model()
            
            masks, features, info = preprocessor.process_image(
                image,
                output_dir=None,
                basename=uploaded_file.name
            )
            
            st.success(f"Detected {info['n_regions']} regions with {info['n_features']} features")
            
            # Step 3: Build graph
            status_text.text("Building graph...")
            progress_bar.progress(60)
            
            constructor = GraphConstructor(
                distance_threshold=distance_threshold,
                k_neighbors=k_neighbors
            )
            
            G = constructor.build_spatial_graph(features, method='knn')
            constructor.add_morphological_edges(G, features)
            
            stats = constructor.get_graph_statistics(G)
            st.success(f"Built graph: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
            
            # Step 4: Visualize
            status_text.text("Generating visualizations...")
            progress_bar.progress(80)
            
            # Create persistent temp directory for visualizations
            if 'viz_dir' not in st.session_state:
                st.session_state['viz_dir'] = tempfile.mkdtemp()
            
            viz_dir = st.session_state['viz_dir']
            visualizer = Visualizer(output_dir=viz_dir)
            
            # Save visualizations
            visualizer.plot_segmentation_overlay(image, masks, save_name="segmentation")
            visualizer.plot_compartment_masks(masks, save_name="compartments")
            visualizer.plot_feature_distributions(features, save_name="features")
            visualizer.plot_graph(G, save_name="graph")
            
            # Store results in session state
            st.session_state['results'] = {
                'metadata': metadata,
                'info': info,
                'stats': stats,
                'features': features,
                'output_dir': viz_dir,
                'image': image,
                'masks': masks
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            st.balloons()
            
            # Show quick summary
            st.markdown("### Quick Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Regions Detected", info['n_regions'])
            with col2:
                st.metric("Features Extracted", info['n_features'])
            with col3:
                st.metric("Graph Nodes", stats['n_nodes'])
            with col4:
                st.metric("Graph Edges", stats['n_edges'])
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        # Clean up temporary TIFF file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


def results_tab():
    """Results display tab."""
    
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.info("👈 Upload and process a TIFF file to see results here")
        return
    
    results = st.session_state['results']
    
    # Metadata
    st.markdown("### Image Metadata")
    metadata_df = pd.DataFrame([results['metadata']]).T
    metadata_df.columns = ['Value']
    st.dataframe(metadata_df, use_container_width=True)
    
    # Processing info
    st.markdown("### Processing Information")
    info_df = pd.DataFrame([results['info']]).T
    info_df.columns = ['Value']
    st.dataframe(info_df, use_container_width=True)
    
    # Graph statistics
    st.markdown("### Graph Statistics")
    stats_df = pd.DataFrame([results['stats']]).T
    stats_df.columns = ['Value']
    st.dataframe(stats_df, use_container_width=True)
    
    # Features table
    st.markdown("### Extracted Features")
    st.dataframe(results['features'], use_container_width=True, height=400)
    
    # Download buttons
    st.markdown("### Download Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = results['features'].to_csv(index=False)
        st.download_button(
            label="📥 Download Features (CSV)",
            data=csv_data,
            file_name="features.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = results['features'].to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download Features (JSON)",
            data=json_data,
            file_name="features.json",
            mime="application/json"
        )
    
    with col3:
        info_json = json.dumps({
            'metadata': results['metadata'],
            'info': results['info'],
            'stats': results['stats']
        }, indent=2)
        st.download_button(
            label="📥 Download Metadata (JSON)",
            data=info_json,
            file_name="metadata.json",
            mime="application/json"
        )


def visualizations_tab():
    """Visualizations display tab."""
    
    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.info("👈 Upload and process a TIFF file to see visualizations here")
        return
    
    results = st.session_state['results']
    output_dir = Path(results['output_dir'])
    
    st.markdown("### Generated Visualizations")
    
    # Display segmentation overlay
    with st.expander("🔬 Segmentation Overlay", expanded=True):
        seg_path = output_dir / "segmentation.png"
        if seg_path.exists():
            st.image(str(seg_path), caption="Segmentation Overlay", use_column_width=True)
        else:
            st.warning("Segmentation visualization not found")
    
    # Display compartment masks
    with st.expander("🎨 Compartment Masks"):
        comp_path = output_dir / "compartments.png"
        if comp_path.exists():
            st.image(str(comp_path), caption="Compartment Masks", use_column_width=True)
        else:
            st.warning("Compartment masks not found")
    
    # Display feature distributions
    with st.expander("📊 Feature Distributions"):
        feat_path = output_dir / "features.png"
        if feat_path.exists():
            st.image(str(feat_path), caption="Feature Distributions", use_column_width=True)
        else:
            st.warning("Feature distributions not found")
    
    # Display graph visualization
    with st.expander("🕸️ Graph Visualization"):
        graph_path = output_dir / "graph.png"
        if graph_path.exists():
            st.image(str(graph_path), caption="Graph Visualization", use_column_width=True)
        else:
            st.warning("Graph visualization not found")
    
    st.markdown("---")
    st.info("💡 All visualizations are publication-ready at 300 DPI")


def help_tab():
    """Help and documentation tab."""
    
    st.markdown('<div class="section-header">Help & Documentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## How to Use This Pipeline
    
    ### 1. Upload TIFF File
    - Click "Browse files" or drag and drop a TIFF file
    - Supports multi-channel, 3D, and 4D TIFF stacks
    
    ### 2. Configure Settings
    Use the sidebar to adjust:
    - **GPU Acceleration**: Enable if CUDA-capable GPU is available
    - **Cell Diameter**: Expected diameter of cells in pixels
    - **Distance Threshold**: Maximum distance for graph edges
    - **K Neighbors**: Number of nearest neighbors in graph
    
    ### 3. Run Pipeline
    Click "Run Pipeline" to start processing. The pipeline will:
    1. Load and validate the TIFF file
    2. Segment cells using Cellpose
    3. Extract morphological and intensity features
    4. Build a biological graph
    5. Generate visualizations
    
    ### 4. View Results
    - **Results Tab**: View metadata, statistics, and feature tables
    - **Visualizations Tab**: Access generated plots and figures
    - Download results as CSV or JSON
    
    ## Features
    
    ### Segmentation
    - Uses Cellpose for accurate cell detection
    - Detects soma, neurites, and protein puncta
    - Handles multi-channel images
    
    ### Feature Extraction
    Extracts comprehensive features:
    - Spatial coordinates
    - Morphological properties (area, perimeter, etc.)
    - Intensity statistics
    - Texture features (GLCM)
    
    ### Graph Construction
    - Builds spatial graphs based on proximity
    - Adds morphological similarity edges
    - Compatible with PyTorch Geometric and DGL
    
    ### Visualization
    - Publication-ready figures
    - Segmentation overlays
    - Feature distributions
    - Graph diagrams
    
    ## System Requirements
    
    - Python 3.8+
    - 4GB+ RAM
    - Optional: CUDA-capable GPU
    
    ## Supported Formats
    
    - `.tif` and `.tiff` files
    - Multi-channel images
    - 3D Z-stacks
    - 4D time series
    
    ## Troubleshooting
    
    **Problem**: "Failed to load TIFF file"
    - Ensure the file is a valid TIFF format
    - Check file is not corrupted
    
    **Problem**: "No regions detected"
    - Adjust cell diameter parameter
    - Check image quality and contrast
    
    **Problem**: "Out of memory"
    - Process smaller images
    - Reduce batch size
    - Use GPU acceleration
    
    ## Contact & Support
    
    For issues and questions:
    - GitHub: https://github.com/soujanyap29/portfolio.github.io
    - Documentation: See `docs/` folder
    - Examples: See `final_pipeline.ipynb`
    """)


if __name__ == "__main__":
    main()
