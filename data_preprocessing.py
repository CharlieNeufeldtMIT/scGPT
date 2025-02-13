import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

def preprocess_data(input_path: str, output_path: str):
    """Loads, normalizes, and prepares gene expression data for training."""
    print(f"Loading dataset from {input_path}...")

    # Detect file format
    if input_path.endswith(".h5ad"):
        print("Detected H5AD file format. Loading with scanpy...")
        adata = sc.read_h5ad(input_path)
    else:  # Assume CSV
        print("Detected CSV file format. Loading with pandas...")
        combined_expression_matrix = pd.read_csv(input_path, index_col=0)
        gene_ids = combined_expression_matrix.index
        expression_data = combined_expression_matrix.values.T
        adata = ad.AnnData(
            X=expression_data,
            var=pd.DataFrame(index=gene_ids),
            obs=pd.DataFrame(index=combined_expression_matrix.columns)
        )

    # Preprocess the data
    print("Normalizing and transforming data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # Save the processed dataset
    adata.write_h5ad(output_path, compression="gzip")
    print(f"Processed data saved at {output_path}")

    # Convert sparse to dense if necessary
    if not isinstance(adata.X, np.ndarray):
        print("Converting sparse matrix to dense...")
        adata.X = adata.X.toarray()  # Convert only if sparse

    print(f"Dataset shape after preprocessing: {adata.X.shape}")

    # Reduce dataset size for memory efficiency (optional)
    subset_size = min(adata.X.shape[0], 2)  # Adjust if needed
    print(f"Using subset of {subset_size} samples for training.")
    train_data, val_data = train_test_split(adata.X[:subset_size], test_size=0.1, random_state=42)

    # Force garbage collection to free memory before tensor conversion
    gc.collect()

    # Convert NumPy arrays to PyTorch tensors
    print("Converting dataset to PyTorch tensors...")

    input_gene_ids_train = torch.arange(train_data.shape[1]).unsqueeze(0).expand(train_data.shape[0], -1)
    input_gene_ids_val = torch.arange(val_data.shape[1]).unsqueeze(0).expand(val_data.shape[0], -1)

    input_values_train = torch.tensor(train_data, dtype=torch.float32)
    target_values_train = torch.tensor(train_data, dtype=torch.float32)

    input_values_val = torch.tensor(val_data, dtype=torch.float32)
    target_values_val = torch.tensor(val_data, dtype=torch.float32)

    # Create DataLoader objects with optimized batch size
    batch_size = 1  # Adjust based on memory constraints
    train_dataset = TensorDataset(input_gene_ids_train, input_values_train, target_values_train)
    val_dataset = TensorDataset(input_gene_ids_val, input_values_val, target_values_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
