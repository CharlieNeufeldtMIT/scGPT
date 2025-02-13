from pathlib import Path

# General settings
DATASET_NAME = "plant_scrna"
SEED = 42
SAVE_DIR = Path("./save/plant_finetuning/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Paths for raw and processed data
RAW_DATA_PATH = "Arabadopsis_root_tissue_1/combined_expression_matrix.h5ad"  
PROCESSED_DATA_PATH = "data/filtered_data.h5ad" 

# Path to the pre-trained scGPT model
PRETRAINED_MODEL_PATH = "saved_models/scGPT_human.pt"

# Hyperparameters
HYPERPARAMS = {
    "do_train": True,
    "load_model": PRETRAINED_MODEL_PATH,  # Use the correct pre-trained model path
    "GEPC": True,  # Masked value prediction (self-supervised learning)
    "ecs_thres": 0.8,  # Elastic cell similarity threshold
    "mask_ratio": 0.4,  # Percentage of values to mask for training
    "epochs": 2,  # Number of training epochs
    "n_bins": 51,  # Number of bins for expression values
    "lr": 0.0001,  # Learning rate
    "batch_size": 64,  # Training batch size
    "layer_size": 128,  # Hidden layer size
    "nlayers": 12,  # Number of Transformer layers
    "nheads": 8,  # Number of attention heads
    "dropout": 0.2,  # Dropout rate
    "schedule_ratio": 0.9,  # Learning rate scheduler ratio
    "embsize": 512,
    "d_hid": 512
}
