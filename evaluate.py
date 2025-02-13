import scanpy as sc

def evaluate(adata):
    """Applies Leiden clustering and evaluates results."""
    print("Performing Leiden clustering...")
    sc.pp.neighbors(adata, n_neighbors=50)
    sc.tl.leiden(adata)
    print("Leiden clustering complete.")
