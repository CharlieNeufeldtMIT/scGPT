import matplotlib.pyplot as plt
import umap
import numpy as np

def plot_umap(embeddings, labels, save_path="results/umap.png"):
    """Generates UMAP visualization."""
    reducer = umap.UMAP(n_components=2, n_neighbors=30)
    umap_embeddings = reducer.fit_transform(embeddings)

    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap="rainbow")
    plt.title("UMAP Visualization")
    plt.savefig(save_path)
    plt.show()
