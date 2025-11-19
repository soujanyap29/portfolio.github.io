"""
Streamlit Web Interface for Protein Sub-Cellular Localization System
"""
import streamlit as st
import sys
import os
import json
from PIL import Image
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from pipeline import ProteinLocalizationPipeline
from config import PROTEIN_CLASSES, INPUT_PATH, OUTPUT_PATH

# Page configuration
st.set_page_config(
    page_title="Protein Sub-Cellular Localization System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .result-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧬 Protein Sub-Cellular Localization in Neurons</div>', 
           unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning and Deep Learning Course Project</div>', 
           unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/2E86AB/FFFFFF?text=Protein+Localization", 
            use_column_width=True)
    st.markdown("---")
    st.markdown("### 📋 System Information")
    st.markdown(f"""
    **Models:**
    - VGG16-based CNN
    - Graph Neural Network (GCN)
    
    **Segmentation:**
    - SLIC Superpixels
    - U-Net (optional)
    - Watershed (optional)
    
    **Classes:**
    {', '.join(PROTEIN_CLASSES[:4])}...
    """)
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("""
    ✓ TIFF Image Upload  
    ✓ Automated Segmentation  
    ✓ Dual Model Classification  
    ✓ Batch Processing  
    ✓ Scientific Visualization  
    ✓ Downloadable Reports
    """)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return ProteinLocalizationPipeline(output_dir=OUTPUT_PATH)

pipeline = get_pipeline()

# Main content
tab1, tab2, tab3 = st.tabs(["🔬 Single Image Analysis", "📁 Batch Processing", "📄 About"])

with tab1:
    st.header("Single Image Analysis")
    st.markdown("Upload a TIFF microscopy image for protein localization analysis.")
    
    uploaded_file = st.file_uploader("Choose a TIFF file", type=['tif', 'tiff'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✓ Uploaded: {uploaded_file.name}")
        
        if st.button("🚀 Analyze Image", type="primary"):
            with st.spinner("Processing image... This may take a few minutes."):
                try:
                    # Process image
                    result = pipeline.process_single_image(temp_path)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("✓ Analysis Complete!")
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        # Create columns for results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("CNN Prediction", 
                                     result['cnn']['predicted_class'],
                                     f"{result['cnn']['confidence']:.1%} confidence")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("GNN Prediction", 
                                     result['gnn']['predicted_class'],
                                     f"{result['gnn']['confidence']:.1%} confidence")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Fused Prediction", 
                                     result['fused']['predicted_class'],
                                     f"{result['fused']['confidence']:.1%} confidence")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Visualizations
                        st.markdown("---")
                        st.subheader("🎨 Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            st.markdown("**Segmentation & Overlay**")
                            if os.path.exists(result['visualizations']['overlay']):
                                st.image(result['visualizations']['overlay'], 
                                        use_column_width=True)
                        
                        with viz_col2:
                            st.markdown("**Probability Distribution**")
                            if os.path.exists(result['visualizations']['probability']):
                                st.image(result['visualizations']['probability'], 
                                        use_column_width=True)
                        
                        # Graph visualization
                        st.markdown("**Graph Network Visualization**")
                        if os.path.exists(result['visualizations']['graph']):
                            st.image(result['visualizations']['graph'], 
                                    use_column_width=True)
                        
                        # Detailed probabilities
                        st.markdown("---")
                        st.subheader("📈 Detailed Probabilities")
                        
                        prob_col1, prob_col2, prob_col3 = st.columns(3)
                        
                        with prob_col1:
                            st.markdown("**CNN Probabilities**")
                            for cls, prob in result['cnn']['probabilities'].items():
                                st.write(f"{cls}: {prob:.3f}")
                        
                        with prob_col2:
                            st.markdown("**GNN Probabilities**")
                            for cls, prob in result['gnn']['probabilities'].items():
                                st.write(f"{cls}: {prob:.3f}")
                        
                        with prob_col3:
                            st.markdown("**Fused Probabilities**")
                            for cls, prob in result['fused']['probabilities'].items():
                                st.write(f"{cls}: {prob:.3f}")
                        
                        # Download report
                        st.markdown("---")
                        report_path = result.get('visualizations', {}).get('segmentation', '').replace('_segment.png', '_report.json')
                        report_path = report_path.replace('/segmented/', '/reports/')
                        
                        if os.path.exists(report_path):
                            with open(report_path, 'r') as f:
                                report_data = f.read()
                            st.download_button(
                                label="📥 Download Full Report (JSON)",
                                data=report_data,
                                file_name=f"{uploaded_file.name}_report.json",
                                mime="application/json"
                            )
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)

with tab2:
    st.header("Batch Processing")
    st.markdown(f"Process all TIFF files in the input directory: `{INPUT_PATH}`")
    
    st.info("⚠️ Batch processing will recursively scan the directory and process all TIFF files.")
    
    if st.button("🚀 Start Batch Processing", type="primary"):
        with st.spinner("Processing batch... This may take a while."):
            try:
                results = pipeline.process_batch(input_dir=INPUT_PATH)
                
                st.success(f"✓ Batch processing complete! Processed {len(results)} images.")
                
                # Display summary
                st.markdown("---")
                st.subheader("📊 Batch Summary")
                
                successful = len([r for r in results if "error" not in r])
                failed = len([r for r in results if "error" in r])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)
                
                # Show results table
                st.markdown("---")
                st.subheader("Results")
                
                for i, result in enumerate(results, 1):
                    if "error" not in result:
                        with st.expander(f"{i}. {result['filename']} - {result['fused']['predicted_class']}"):
                            st.write(f"**Confidence:** {result['fused']['confidence']:.1%}")
                            st.write(f"**CNN:** {result['cnn']['predicted_class']}")
                            st.write(f"**GNN:** {result['gnn']['predicted_class']}")
                    else:
                        with st.expander(f"{i}. {result.get('filename', 'Unknown')} - ERROR"):
                            st.error(result['error'])
            
            except Exception as e:
                st.error(f"An error occurred during batch processing: {str(e)}")
                st.exception(e)

with tab3:
    st.header("About This System")
    
    st.markdown("""
    ## Protein Sub-Cellular Localization in Neurons
    
    ### Course Information
    **Course:** Machine Learning and Deep Learning  
    **Project Type:** Complete Scientific System
    
    ### System Overview
    This system provides automated analysis of neuronal TIFF microscopy images for 
    protein sub-cellular localization classification using state-of-the-art deep learning techniques.
    
    ### Key Features
    
    #### 🧠 Machine Learning Models
    1. **VGG16-based Deep CNN**
       - Fine-tuned on neuronal microscopy datasets
       - Global feature extraction
       - Transfer learning from ImageNet
    
    2. **Graph Neural Network (GNN)**
       - Superpixel-based graph construction
       - Node features: intensity, texture, geometry
       - Architectures: GCN, GraphSAGE, GAT
       - Captures spatial relationships
    
    #### 🧬 Segmentation Methods
    - SLIC Superpixel Segmentation
    - U-Net Deep Learning Segmentation
    - Watershed Segmentation
    
    #### 📊 Evaluation Metrics
    - Accuracy, Precision, Recall
    - F1-Score, Specificity
    - Confusion Matrix
    - Probability Distributions
    
    #### 🎨 Scientific Visualizations
    - High-resolution outputs (300+ DPI)
    - Segmentation overlays
    - Probability distributions
    - Graph network visualizations
    - Publication-ready quality
    
    ### Protein Localization Classes
    """)
    
    for i, cls in enumerate(PROTEIN_CLASSES, 1):
        st.markdown(f"{i}. **{cls}**")
    
    st.markdown("""
    ### System Architecture
    
    ```
    Input TIFF Image
         ↓
    Preprocessing & Normalization
         ↓
    Segmentation (SLIC/U-Net/Watershed)
         ↓
    ├─→ VGG16 CNN ────────────┐
    │                         ↓
    └─→ Superpixel Graph → GNN ──→ Model Fusion → Final Prediction
                                         ↓
                              Visualization & Report Generation
    ```
    
    ### Output Structure
    ```
    /output
        /results
            /segmented     - Segmentation masks
            /predictions   - Classification results
            /reports       - JSON reports
        /graphs           - Scientific visualizations
    ```
    
    ### Technical Stack
    - **Deep Learning:** TensorFlow, PyTorch
    - **Graph Learning:** PyTorch Geometric
    - **Image Processing:** scikit-image, OpenCV
    - **Visualization:** Matplotlib, Seaborn
    - **Web Interface:** Streamlit
    
    ### Applications
    - Neurodegenerative disease research
    - Synaptic protein mapping
    - Drug discovery
    - Cell-type classification
    - Biomarker studies
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <p>Protein Sub-Cellular Localization System | Machine Learning and Deep Learning Course</p>
    <p>© 2025 | Built with Streamlit, TensorFlow, and PyTorch</p>
</div>
""", unsafe_allow_html=True)
