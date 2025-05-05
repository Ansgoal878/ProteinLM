"""
Visualization module for protein embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap 
import pandas as pd

class EmbeddingVisualizer:
    def __init__(self, embeddings, labels=None):
        """
        Initialize visualizer with embeddings
        
        Args:
            embeddings (numpy.ndarray): Matrix of embeddings
            labels (list): Optional labels for each embedding
        """
        self.embeddings = embeddings
        self.labels = labels or np.arange(len(embeddings))
        
    def run_pca(self, n_components=2):
        """
        Perform PCA on embeddings
        
        Args:
            n_components (int): Number of components
            
        Returns:
            numpy.ndarray: PCA-reduced embeddings
        """
        pca = PCA(n_components=n_components)
        return pca.fit_transform(self.embeddings)
    
    def run_tsne(self, n_components=2, perplexity=30):
        """
        Perform t-SNE on embeddings
        
        Args:
            n_components (int): Number of components
            perplexity (float): t-SNE perplexity parameter
            
        Returns:
            numpy.ndarray: t-SNE-reduced embeddings
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        return tsne.fit_transform(self.embeddings)
    
    def run_umap(self, n_components=2, min_dist=0.1, n_neighbors=15):
        """
        Perform UMAP on embeddings
        
        Args:
            n_components (int): Number of components
            min_dist (float): UMAP min_dist parameter
            n_neighbors (int): UMAP n_neighbors parameter
            
        Returns:
            numpy.ndarray: UMAP-reduced embeddings
        """
        # Get sample size
        sample_size = self.embeddings.shape[0]
        
        # Adjust n_neighbors based on sample size
        adjusted_n_neighbors = min(n_neighbors, sample_size - 1)
        
        # For small datasets, further reduce n_neighbors and min_dist
        if sample_size < 20:
            # Smaller n_neighbors for small datasets
            suggested_n_neighbors = max(2, sample_size // 2)
            adjusted_n_neighbors = min(adjusted_n_neighbors, suggested_n_neighbors)
            
            # Smaller min_dist for small datasets to avoid spreading points too far
            min_dist = min(min_dist, 0.05)
            
            print(f"Small dataset detected (size {sample_size}). Adjusted UMAP parameters:")
            print(f"  - n_neighbors: {adjusted_n_neighbors} (from original {n_neighbors})")
            print(f"  - min_dist: {min_dist}")
        
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=adjusted_n_neighbors,
                min_dist=min_dist,
                metric='euclidean',
                random_state=42
            )
            
            # Add a small amount of noise for very small datasets to help UMAP
            if sample_size < 15:
                import numpy as np
                # Add tiny noise to prevent singularity issues
                noise_scale = 1e-4
                noise = np.random.normal(0, noise_scale, self.embeddings.shape)
                embeddings_with_noise = self.embeddings + noise
                umap_embeddings = reducer.fit_transform(embeddings_with_noise)
            else:
                umap_embeddings = reducer.fit_transform(self.embeddings)
                
            return umap_embeddings
        except Exception as e:
            print(f"UMAP error: {str(e)}. Falling back to PCA.")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            return pca.fit_transform(self.embeddings)
    
    def plot_2d(self, embeddings_2d, labels=None, title="Protein Embeddings", 
                save_path=None, figsize=(10, 8), point_size=50, method="UMAP"):
        """
        Plot 2D embeddings
        
        Args:
            embeddings_2d (numpy.ndarray): 2D embeddings to plot
            labels (list): Labels for color coding
            title (str): Plot title
            save_path (str): Path to save figure
            figsize (tuple): Figure size
            point_size (int): Size of scatter points
            method (str): Method used for dimensionality reduction
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=figsize)
        
        # Use labels if provided, otherwise use default
        plot_labels = labels if labels is not None else self.labels
        
        # Create scatter plot
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=plot_labels if isinstance(plot_labels[0], (int, float)) else None,
                            s=point_size, alpha=0.7)
        
        # Add legend if categorical labels
        if not isinstance(plot_labels[0], (int, float)):
            for i, label in enumerate(np.unique(plot_labels)):
                idx = np.where(np.array(plot_labels) == label)[0]
                plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                        label=label, s=point_size, alpha=0.7)
            plt.legend()
        else:
            plt.colorbar(scatter)
        
        plt.title(f"{title} - {method}")
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")
        plt.tight_layout()
        
        # Generate a default save path if none provided
        if save_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"protein_embeddings_{method.lower()}_{timestamp}.png"
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        
        plt.close()