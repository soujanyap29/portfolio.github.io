"""
Graph visualization for protein localization predictions
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Optional
import seaborn as sns


class GraphVisualizer:
    """Visualize graphs with node labels and predictions"""
    
    def __init__(self, figsize=(12, 10)):
        """
        Initialize visualizer
        
        Args:
            figsize: Figure size tuple
        """
        self.figsize = figsize
        self.color_map = {
            'nucleus': '#FF6B6B',
            'mitochondria': '#4ECDC4',
            'endoplasmic_reticulum': '#45B7D1',
            'golgi': '#FFA07A',
            'cytoplasm': '#98D8C8',
            'membrane': '#F7DC6F',
            'unknown': '#95A5A6'
        }
    
    def visualize_graph(self, G: nx.Graph, 
                       predictions: Optional[Dict[int, str]] = None,
                       save_path: Optional[str] = None,
                       title: str = "Protein Sub-Cellular Localization Graph"):
        """
        Visualize graph with node labels
        
        Args:
            G: NetworkX graph
            predictions: Dictionary mapping node IDs to predicted classes
            save_path: Optional path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get node positions
        pos = nx.spring_layout(G, seed=42)
        
        # If positions are stored in node attributes, use them
        if G.nodes() and 'pos' in G.nodes[list(G.nodes())[0]]:
            pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        
        # Determine node colors
        if predictions:
            node_colors = [self.color_map.get(predictions.get(node, 'unknown'), 
                                             self.color_map['unknown']) 
                          for node in G.nodes()]
        else:
            # Use class_label if available
            node_colors = []
            for node in G.nodes():
                label = G.nodes[node].get('class_label', 'unknown')
                node_colors.append(self.color_map.get(label, self.color_map['unknown']))
        
        # Get node sizes based on area
        node_sizes = []
        for node in G.nodes():
            area = G.nodes[node].get('area', 300)
            node_sizes.append(max(300, min(area, 2000)))
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)
        
        nx.draw_networkx_edges(G, pos, 
                              alpha=0.3,
                              width=2,
                              ax=ax)
        
        # Draw labels
        labels = {}
        for node in G.nodes():
            if predictions and node in predictions:
                labels[node] = f"{node}\n{predictions[node]}"
            elif 'class_label' in G.nodes[node]:
                labels[node] = f"{node}\n{G.nodes[node]['class_label']}"
            else:
                labels[node] = str(node)
        
        nx.draw_networkx_labels(G, pos, labels, 
                               font_size=8,
                               font_weight='bold',
                               ax=ax)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=color, markersize=10,
                                     label=label.replace('_', ' ').title())
                          for label, color in self.color_map.items()
                          if label != 'unknown']
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=10, framealpha=0.9)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {save_path}")
        
        return fig
    
    def visualize_features(self, feature_matrix: np.ndarray,
                          labels: Optional[list] = None,
                          feature_names: Optional[list] = None,
                          save_path: Optional[str] = None):
        """
        Visualize node features as heatmap
        
        Args:
            feature_matrix: Matrix of features (nodes x features)
            labels: Node labels
            feature_names: Names of features
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(feature_matrix.shape[1])]
        
        sns.heatmap(feature_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd',
                   xticklabels=feature_names,
                   yticklabels=labels if labels else range(feature_matrix.shape[0]),
                   cbar_kws={'label': 'Feature Value'},
                   ax=ax)
        
        ax.set_title('Node Feature Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Nodes', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature heatmap saved to {save_path}")
        
        return fig
    
    def plot_training_history(self, history: Dict,
                              save_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            history: Dictionary with 'train_loss' and 'test_accuracy'
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], linewidth=2, color='#3498db')
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history['test_accuracy'], linewidth=2, color='#2ecc71')
        ax2.set_title('Test Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, 
                             class_names: list,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Optional path to save figure
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax)
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def visualize_biological_network(self, G: nx.Graph,
                                     central_node: Optional[int] = None,
                                     save_path: Optional[str] = None,
                                     title: str = "Biological Network Diagram"):
        """
        Generate a clean biological network diagram with scientific aesthetic
        
        Args:
            G: NetworkX graph
            central_node: Node ID to highlight as central (blue)
            save_path: Optional path to save figure
            title: Plot title
        """
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, ConnectionPatch
        from matplotlib.collections import PatchCollection
        
        # Create figure with light grey background
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='#F5F5F5')
        ax.set_facecolor('#F5F5F5')
        
        # Get node positions using force-directed layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # If positions are stored in node attributes, use them
        if G.nodes() and 'pos' in G.nodes[list(G.nodes())[0]]:
            pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        
        # Normalize positions to fill the space nicely
        pos_array = np.array(list(pos.values()))
        pos_min = pos_array.min(axis=0)
        pos_max = pos_array.max(axis=0)
        pos_range = pos_max - pos_min
        
        for node in pos:
            pos[node] = ((pos[node] - pos_min) / pos_range) * 0.8 + 0.1
        
        # Determine central node if not specified
        if central_node is None:
            # Choose node with highest degree
            degrees = dict(G.degree())
            central_node = max(degrees, key=degrees.get) if degrees else list(G.nodes())[0]
        
        # Identify clusters (communities) for grouping
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G))
        except:
            # Fallback: simple clustering based on connectivity
            communities = [{node} for node in G.nodes()]
        
        # Draw translucent cluster backgrounds
        for i, comm in enumerate(communities):
            if len(comm) > 2:  # Only draw for groups of 3+ nodes
                comm_pos = np.array([pos[node] for node in comm])
                
                # Calculate convex hull
                from scipy.spatial import ConvexHull
                if len(comm_pos) >= 3:
                    try:
                        hull = ConvexHull(comm_pos)
                        hull_points = comm_pos[hull.vertices]
                        
                        # Expand hull slightly
                        center = comm_pos.mean(axis=0)
                        expanded_points = center + (hull_points - center) * 1.3
                        
                        cluster_patch = plt.Polygon(
                            expanded_points,
                            facecolor='#E8E8E8',
                            edgecolor='none',
                            alpha=0.3,
                            zorder=1
                        )
                        ax.add_patch(cluster_patch)
                    except:
                        pass
        
        # Draw edges with curved lines
        for edge in G.edges():
            node1, node2 = edge
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]
            
            # Calculate control point for curved line
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            # Perpendicular offset
            offset = 0.05
            ctrl_x = mid_x - dy * offset
            ctrl_y = mid_y + dx * offset
            
            # Draw curved edge using bezier curve
            from matplotlib.path import Path
            import matplotlib.patches as patches
            
            verts = [
                (x1, y1),
                (ctrl_x, ctrl_y),
                (x2, y2),
            ]
            
            codes = [
                Path.MOVETO,
                Path.CURVE3,
                Path.CURVE3,
            ]
            
            path = Path(verts, codes)
            patch = patches.PathPatch(
                path,
                facecolor='none',
                edgecolor='#CCCCCC',
                linewidth=1.5,
                alpha=0.6,
                zorder=2
            )
            ax.add_patch(patch)
        
        # Draw nodes as rounded rectangles
        for node in G.nodes():
            x, y = pos[node]
            
            # Determine if this is the central node
            is_central = (node == central_node)
            
            # Node styling
            if is_central:
                facecolor = '#4A90E2'  # Distinct blue
                edgecolor = '#3A7BC8'
                alpha = 1.0
            else:
                facecolor = '#D3D3D3'  # Soft grey
                edgecolor = '#B8B8B8'
                alpha = 0.9
            
            # Create rounded rectangle
            width, height = 0.08, 0.05
            rect = FancyBboxPatch(
                (x - width/2, y - height/2),
                width,
                height,
                boxstyle="round,pad=0.01",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.5,
                alpha=alpha,
                zorder=3,
                # Add soft shadow
                path_effects=[
                    plt.matplotlib.patheffects.withStroke(
                        linewidth=3,
                        foreground='#00000020'
                    )
                ]
            )
            ax.add_patch(rect)
            
            # Add node label
            label_text = str(node)
            if 'class_label' in G.nodes[node]:
                label_text = G.nodes[node]['class_label']
            
            ax.text(
                x, y,
                label_text,
                fontsize=7,
                fontweight='bold' if is_central else 'normal',
                ha='center',
                va='center',
                color='white' if is_central else '#333333',
                zorder=4
            )
        
        # Set axis properties
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axis('off')
        
        # Add title
        ax.text(
            0.5, 0.98,
            title,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='top',
            color='#333333'
        )
        
        # Add subtitle/legend
        legend_text = (
            f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
            f"Blue node: Central hub | Grey nodes: Network members"
        )
        ax.text(
            0.5, 0.02,
            legend_text,
            transform=ax.transAxes,
            fontsize=9,
            ha='center',
            va='bottom',
            color='#666666',
            style='italic'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#F5F5F5')
            print(f"Biological network diagram saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    print("Testing graph visualization...")
    
    # Create test graph
    G = nx.Graph()
    
    # Add nodes with features
    for i in range(6):
        G.add_node(i, 
                  pos=(np.random.rand() * 10, np.random.rand() * 10),
                  area=np.random.randint(200, 800))
    
    # Add edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
    G.add_edges_from(edges)
    
    # Add predictions
    predictions = {
        0: 'nucleus',
        1: 'mitochondria',
        2: 'endoplasmic_reticulum',
        3: 'golgi',
        4: 'cytoplasm',
        5: 'membrane'
    }
    
    # Visualize
    visualizer = GraphVisualizer()
    fig = visualizer.visualize_graph(G, predictions)
    
    # Don't show plot in non-interactive environment
    print("Graph visualization created successfully!")
    plt.close()
    
    # Test feature visualization
    feature_matrix = np.random.rand(6, 4)
    fig = visualizer.visualize_features(feature_matrix, 
                                       labels=[f"Node {i}" for i in range(6)],
                                       feature_names=['Area', 'Intensity', 
                                                    'Eccentricity', 'Solidity'])
    plt.close()
    
    print("Feature visualization created successfully!")
