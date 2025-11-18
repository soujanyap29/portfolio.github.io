"""
Graph visualization module
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


class GraphVisualizer:
    """Visualize graphs with clean styling"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_graph(self,
                   G: nx.Graph,
                   node_labels: Optional[Dict] = None,
                   node_colors: Optional[List] = None,
                   node_sizes: Optional[List] = None,
                   title: str = "Graph Visualization",
                   filename: str = None,
                   layout: str = 'spring'):
        """
        Plot graph with rounded nodes and clean styling
        
        Args:
            G: NetworkX graph
            node_labels: Dictionary of node labels
            node_colors: List of node colors
            node_sizes: List of node sizes
            title: Plot title
            filename: Output filename
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Default node properties
        if node_colors is None:
            node_colors = ['#1f77b4'] * G.number_of_nodes()
        
        if node_sizes is None:
            node_sizes = [300] * G.number_of_nodes()
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1.0,
            alpha=0.6,
            ax=ax
        )
        
        # Draw nodes with rounded appearance
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=2,
            ax=ax
        )
        
        # Draw labels
        if node_labels:
            nx.draw_networkx_labels(
                G, pos,
                node_labels,
                font_size=8,
                font_weight='bold',
                font_color='white',
                ax=ax
            )
        else:
            # Use node IDs as labels
            labels = {node: str(node) for node in G.nodes()}
            nx.draw_networkx_labels(
                G, pos,
                labels,
                font_size=8,
                font_weight='bold',
                font_color='white',
                ax=ax
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_bipartite_graph(self,
                            G: nx.Graph,
                            puncta_nodes: List,
                            compartment_nodes: List,
                            title: str = "Bipartite Graph",
                            filename: str = None):
        """
        Plot bipartite graph with two node sets
        
        Args:
            G: NetworkX bipartite graph
            puncta_nodes: List of puncta node IDs
            compartment_nodes: List of compartment node IDs
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create bipartite layout
        pos = {}
        
        # Position puncta nodes on left
        puncta_y = np.linspace(0, 1, len(puncta_nodes))
        for i, node in enumerate(puncta_nodes):
            pos[node] = (0, puncta_y[i])
        
        # Position compartment nodes on right
        comp_y = np.linspace(0, 1, len(compartment_nodes))
        for i, node in enumerate(compartment_nodes):
            pos[node] = (1, comp_y[i])
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1.5,
            alpha=0.6,
            ax=ax
        )
        
        # Draw puncta nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=puncta_nodes,
            node_color='#ff7f0e',
            node_size=500,
            alpha=0.9,
            edgecolors='white',
            linewidths=2,
            label='Puncta',
            ax=ax
        )
        
        # Draw compartment nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=compartment_nodes,
            node_color='#2ca02c',
            node_size=800,
            alpha=0.9,
            edgecolors='white',
            linewidths=2,
            label='Compartments',
            ax=ax
        )
        
        # Draw labels
        puncta_labels = {node: f'P{i}' for i, node in enumerate(puncta_nodes)}
        comp_labels = {node: f'C{i}' for i, node in enumerate(compartment_nodes)}
        all_labels = {**puncta_labels, **comp_labels}
        
        nx.draw_networkx_labels(
            G, pos,
            all_labels,
            font_size=8,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        ax.axis('off')
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_graph_with_features(self,
                                G: nx.Graph,
                                feature_name: str,
                                title: str = "Graph with Features",
                                filename: str = None,
                                cmap: str = 'viridis'):
        """
        Plot graph colored by node feature values
        
        Args:
            G: NetworkX graph
            feature_name: Name of feature to visualize
            title: Plot title
            filename: Output filename
            cmap: Colormap
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract feature values
        feature_values = []
        for node in G.nodes():
            if feature_name in G.nodes[node]:
                feature_values.append(G.nodes[node][feature_name])
            else:
                feature_values.append(0)
        
        # Normalize feature values for coloring
        feature_values = np.array(feature_values)
        if feature_values.max() > feature_values.min():
            normalized_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
        else:
            normalized_values = np.ones_like(feature_values)
        
        # Create colormap
        from matplotlib import cm
        colormap = cm.get_cmap(cmap)
        node_colors = [colormap(val) for val in normalized_values]
        
        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.0, alpha=0.6, ax=ax)
        
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.9,
            edgecolors='white',
            linewidths=2,
            ax=ax
        )
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=colormap, 
                              norm=plt.Normalize(vmin=feature_values.min(), 
                                               vmax=feature_values.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(feature_name, rotation=270, labelpad=20)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Saved: {filepath}")
        
        plt.close()
    
    def plot_graph_statistics(self,
                             G: nx.Graph,
                             title: str = "Graph Statistics",
                             filename: str = None):
        """
        Plot graph statistics
        
        Args:
            G: NetworkX graph
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Degree distribution
        degrees = [G.degree(n) for n in G.nodes()]
        axes[0, 0].hist(degrees, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Degree')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Degree Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Clustering coefficient distribution
        clustering = list(nx.clustering(G).values())
        axes[0, 1].hist(clustering, bins=20, color='#ff7f0e', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Clustering Coefficient')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Clustering Coefficient Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Node degree vs clustering coefficient
        degrees_list = [G.degree(n) for n in G.nodes()]
        clustering_list = [nx.clustering(G, n) for n in G.nodes()]
        axes[1, 0].scatter(degrees_list, clustering_list, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Degree')
        axes[1, 0].set_ylabel('Clustering Coefficient')
        axes[1, 0].set_title('Degree vs Clustering')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Graph summary statistics
        stats_text = f"""
        Nodes: {G.number_of_nodes()}
        Edges: {G.number_of_edges()}
        Avg Degree: {np.mean(degrees):.2f}
        Density: {nx.density(G):.4f}
        Avg Clustering: {np.mean(clustering):.4f}
        """
        
        if nx.is_connected(G):
            stats_text += f"\nDiameter: {nx.diameter(G)}"
            stats_text += f"\nAvg Path Length: {nx.average_shortest_path_length(G):.2f}"
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
        
        plt.close()


if __name__ == "__main__":
    print("Graph visualization module ready!")
    
    # Test with a simple graph
    G = nx.karate_club_graph()
    visualizer = GraphVisualizer()
    visualizer.plot_graph(G, title="Test Graph")
    print("Test graph created successfully!")
