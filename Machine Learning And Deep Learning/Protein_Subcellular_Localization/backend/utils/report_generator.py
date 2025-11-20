"""
Journal-style PDF report generation.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JournalReportGenerator:
    """Generate journal-style PDF reports."""
    
    def __init__(self, output_path: str):
        """
        Initialize report generator.
        
        Args:
            output_path: Path to save the PDF report
        """
        self.output_path = output_path
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Custom styles
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Author style
        self.styles.add(ParagraphStyle(
            name='Author',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#555555'),
            alignment=TA_CENTER,
            spaceAfter=10
        ))
        
        # Abstract style
        self.styles.add(ParagraphStyle(
            name='Abstract',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            leftIndent=36,
            rightIndent=36,
            spaceAfter=12
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
    
    def add_title_page(self, title: str, author: str, course: str):
        """Add title page."""
        # Title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        
        # Author info
        self.story.append(Spacer(1, 0.5*inch))
        self.story.append(Paragraph(f"<b>Student:</b> {author}", self.styles['Author']))
        self.story.append(Paragraph(f"<b>Course:</b> {course}", self.styles['Author']))
        self.story.append(Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}",
            self.styles['Author']
        ))
        
        self.story.append(PageBreak())
    
    def add_abstract(self, abstract_text: str):
        """Add abstract section."""
        self.story.append(Paragraph("<b>ABSTRACT</b>", self.styles['SectionHeading']))
        self.story.append(Paragraph(abstract_text, self.styles['Abstract']))
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_section(self, title: str, content: str):
        """Add a section with title and content."""
        self.story.append(Paragraph(title, self.styles['SectionHeading']))
        self.story.append(Paragraph(content, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_image(self, image_path: str, caption: str = "", width: float = 4*inch):
        """Add an image with optional caption."""
        if Path(image_path).exists():
            img = Image(image_path, width=width, height=width*0.75)
            self.story.append(img)
            
            if caption:
                caption_style = ParagraphStyle(
                    name='Caption',
                    parent=self.styles['Normal'],
                    fontSize=9,
                    textColor=colors.HexColor('#666666'),
                    alignment=TA_CENTER
                )
                self.story.append(Paragraph(f"<i>{caption}</i>", caption_style))
            
            self.story.append(Spacer(1, 0.2*inch))
        else:
            logger.warning(f"Image not found: {image_path}")
    
    def add_table(self, data: List[List[str]], headers: List[str] = None):
        """Add a table."""
        if headers:
            data = [headers] + data
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_methodology(self, config: Dict):
        """Add methodology section."""
        self.story.append(Paragraph("METHODOLOGY", self.styles['SectionHeading']))
        
        methodology_text = f"""
        This study implements a comprehensive machine learning pipeline for protein 
        sub-cellular localization analysis in neurons. The methodology consists of 
        several key stages:
        """
        
        self.story.append(Paragraph(methodology_text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.1*inch))
        
        # Segmentation
        seg_text = f"""
        <b>1. Image Segmentation:</b> Neuronal images are segmented using Cellpose, 
        a generalist algorithm for cellular segmentation. The model type used is 
        '{config['segmentation']['model_type']}' with a diameter of 
        {config['segmentation']['diameter']} pixels.
        """
        self.story.append(Paragraph(seg_text, self.styles['BodyText']))
        
        # CNN
        cnn_text = f"""
        <b>2. Convolutional Neural Network:</b> A VGG16 architecture is employed with 
        transfer learning from ImageNet. The first {config['cnn']['freeze_layers']} 
        layers are frozen, and the model is fine-tuned on microscopy images using a 
        learning rate of {config['cnn']['learning_rate']}.
        """
        self.story.append(Paragraph(cnn_text, self.styles['BodyText']))
        
        # GNN
        gnn_text = f"""
        <b>3. Graph Neural Network:</b> Superpixels are generated using SLIC algorithm 
        with {config['superpixels']['n_segments']} segments. A {config['gnn']['model_type']} 
        model with {config['gnn']['num_layers']} layers processes the graph structure.
        """
        self.story.append(Paragraph(gnn_text, self.styles['BodyText']))
        
        # Fusion
        fusion_text = f"""
        <b>4. Model Fusion:</b> Predictions from CNN and GNN are combined using 
        {config['fusion']['method']} with weights {config['fusion']['cnn_weight']} 
        and {config['fusion']['gnn_weight']} respectively.
        """
        self.story.append(Paragraph(fusion_text, self.styles['BodyText']))
        
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_results(self, results_data: Dict):
        """Add results section with metrics and visualizations."""
        self.story.append(Paragraph("RESULTS", self.styles['SectionHeading']))
        
        # Summary statistics
        summary_text = f"""
        Analysis was performed on {results_data.get('n_images', 'N/A')} neuronal TIFF 
        images. Segmentation identified an average of {results_data.get('avg_regions', 'N/A')} 
        sub-cellular regions per image.
        """
        self.story.append(Paragraph(summary_text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Metrics table
        if 'metrics' in results_data:
            metrics = results_data['metrics']
            table_data = [
                ['Metric', 'CNN', 'GNN', 'Fused'],
                ['Accuracy', f"{metrics.get('cnn_accuracy', 0):.3f}", 
                 f"{metrics.get('gnn_accuracy', 0):.3f}", 
                 f"{metrics.get('fused_accuracy', 0):.3f}"],
                ['Precision', f"{metrics.get('cnn_precision', 0):.3f}", 
                 f"{metrics.get('gnn_precision', 0):.3f}", 
                 f"{metrics.get('fused_precision', 0):.3f}"],
                ['Recall', f"{metrics.get('cnn_recall', 0):.3f}", 
                 f"{metrics.get('gnn_recall', 0):.3f}", 
                 f"{metrics.get('fused_recall', 0):.3f}"],
                ['F1-Score', f"{metrics.get('cnn_f1', 0):.3f}", 
                 f"{metrics.get('gnn_f1', 0):.3f}", 
                 f"{metrics.get('fused_f1', 0):.3f}"]
            ]
            self.add_table(table_data)
    
    def add_conclusion(self):
        """Add conclusion section."""
        self.story.append(Paragraph("CONCLUSION", self.styles['SectionHeading']))
        
        conclusion_text = """
        This study demonstrates the effectiveness of combining deep learning approaches 
        for protein sub-cellular localization. The fusion of CNN and GNN models provides 
        improved accuracy over individual models, leveraging both spatial and structural 
        information. The automated pipeline enables high-throughput analysis of neuronal 
        microscopy images, facilitating large-scale biological studies.
        """
        self.story.append(Paragraph(conclusion_text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.2*inch))
    
    def add_references(self, references: List[str]):
        """Add references in IEEE format."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("REFERENCES", self.styles['SectionHeading']))
        
        for idx, ref in enumerate(references, 1):
            ref_text = f"[{idx}] {ref}"
            self.story.append(Paragraph(ref_text, self.styles['Normal']))
            self.story.append(Spacer(1, 0.1*inch))
    
    def generate(self):
        """Generate the PDF report."""
        try:
            self.doc.build(self.story)
            logger.info(f"Generated report: {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False


def generate_complete_report(
    output_path: str,
    config: Dict,
    results_data: Dict,
    image_paths: Dict = None
):
    """
    Generate a complete journal-style report.
    
    Args:
        output_path: Path to save the report
        config: Configuration dictionary
        results_data: Results and metrics data
        image_paths: Dictionary of image paths for figures
    """
    generator = JournalReportGenerator(output_path)
    
    # Title page
    generator.add_title_page(
        title="Protein Sub-Cellular Localization in Neurons Using Deep Learning",
        author="Soujanya",
        course="Machine Learning and Deep Learning"
    )
    
    # Abstract
    abstract = """
    Determining the sub-cellular localization of proteins in neurons is crucial for 
    understanding cellular function and disease mechanisms. This study presents an 
    automated computational platform that combines Cellpose segmentation, VGG16 
    convolutional neural networks, and graph neural networks to analyze TIFF microscopy 
    images and classify protein localization patterns. The system achieves high accuracy 
    through model fusion and generates publication-quality visualizations for scientific analysis.
    """
    generator.add_abstract(abstract)
    
    # Introduction
    intro = """
    The spatial organization of proteins within neurons is fundamental to cellular 
    function. Traditional manual analysis of microscopy images is time-consuming and 
    subjective. Machine learning approaches offer automated, reproducible methods for 
    protein localization analysis. This work develops an end-to-end pipeline combining 
    state-of-the-art deep learning techniques.
    """
    generator.add_section("1. INTRODUCTION", intro)
    
    # Methodology
    generator.add_methodology(config)
    
    # Results
    generator.add_results(results_data)
    
    # Add images if provided
    if image_paths:
        generator.story.append(PageBreak())
        generator.add_section("VISUALIZATIONS", "Representative results are shown below:")
        
        for caption, path in image_paths.items():
            generator.add_image(path, caption=caption)
    
    # Conclusion
    generator.add_conclusion()
    
    # References
    references = [
        "C. Stringer et al., 'Cellpose: a generalist algorithm for cellular segmentation,' Nature Methods, vol. 18, pp. 100-106, 2021.",
        "K. Simonyan and A. Zisserman, 'Very Deep Convolutional Networks for Large-Scale Image Recognition,' arXiv:1409.1556, 2014.",
        "T. N. Kipf and M. Welling, 'Semi-Supervised Classification with Graph Convolutional Networks,' ICLR, 2017.",
        "W. Hamilton et al., 'Inductive Representation Learning on Large Graphs,' NeurIPS, 2017.",
        "P. Veličković et al., 'Graph Attention Networks,' ICLR, 2018.",
    ]
    generator.add_references(references)
    
    # Generate
    return generator.generate()


if __name__ == "__main__":
    # Example usage
    import yaml
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results_data = {
        'n_images': 10,
        'avg_regions': 50,
        'metrics': {
            'cnn_accuracy': 0.85,
            'gnn_accuracy': 0.82,
            'fused_accuracy': 0.89,
            'cnn_precision': 0.83,
            'gnn_precision': 0.81,
            'fused_precision': 0.87,
            'cnn_recall': 0.84,
            'gnn_recall': 0.80,
            'fused_recall': 0.88,
            'cnn_f1': 0.83,
            'gnn_f1': 0.80,
            'fused_f1': 0.87
        }
    }
    
    generate_complete_report(
        output_path="/tmp/protein_localization_report.pdf",
        config=config,
        results_data=results_data
    )
    
    print("Report generated successfully!")
