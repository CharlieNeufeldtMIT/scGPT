import torch
import os
import json
from pathlib import Path
import wandb
from models import initialize_model
from config import HYPERPARAMS, RAW_DATA_PATH, PROCESSED_DATA_PATH
from data_preprocessing import preprocess_data
from evaluate import evaluate
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from tqdm import tqdm

# # Select device: Use MPS for Apple Silicon, else fallback to CPU
# if torch.backends.mps.is_available():
#     device = torch.device("mps")  # Apple Metal acceleration
#     print("Using MPS (Apple Metal) for training")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")  # CUDA for NVIDIA GPUs
#     print("Using CUDA for training")
# else:
#     device = torch.device("cpu")  # Default to CPU
#     print("Using CPU for training")

device = torch.device("cpu")  # Force CPU instead of MPS

# Define model directory and file paths
model_dir = Path("pretrained_model")
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

# Check if required files exist
if not model_file.exists():
    print(f"Error: Pre-trained model not found at {model_file}")
    exit(1)
if not model_config_file.exists():
    print(f"Error: Model config file not found at {model_config_file}")
    exit(1)
if not vocab_file.exists():
    print(f"Error: Vocabulary file not found at {vocab_file}")
    exit(1)

# Load vocabulary
vocab = GeneVocab.from_file(vocab_file)

# Load model configurations from args.json
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
print(f"Resuming model from {model_file}, using config {model_config_file}")

 # Initialize the model
model = initialize_model(vocab, pretrained_path=model_dir)
model.to(device)

def train(train_loader, val_loader):
    """Trains the Transformer model using a pre-trained scGPT checkpoint if available."""
    global model # use preloaded model
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"])
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{HYPERPARAMS['epochs']}", leave=True)

        for batch, (input_gene_ids, input_values, target_values) in enumerate(progress_bar):
            input_gene_ids, input_values, target_values = (
                input_gene_ids.to(device),
                input_values.to(device),
                target_values.to(device),
            )

            optimizer.zero_grad()
            output_dict = model(input_gene_ids, input_values, src_key_padding_mask=None)
            loss = criterion(output_dict["mlm_output"], target_values)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        val_loss = evaluate(model, val_loader)

        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"best_model_epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (Epoch {epoch}) to {save_path}")

        print(f"Epoch {epoch}: Training Loss = {total_loss:.4f}, Validation Loss = {val_loss:.4f}")

if __name__ == "__main__":
    train_loader, val_loader = preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    train(train_loader, val_loader)