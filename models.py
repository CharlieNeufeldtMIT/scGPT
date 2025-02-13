import json
import os
import torch
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from config import HYPERPARAMS

def initialize_model(vocab, pretrained_path=None):
    """Loads a pre-trained scGPT model or initializes a new one."""

    # Handle cases where pretrained_path is a file instead of a directory
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Detected model checkpoint: {pretrained_path}")
        model_file = pretrained_path  # Use file directly
        pretrained_path = None  # No directory structure
    else:
        model_file = f"{pretrained_path}/best_model.pt" if pretrained_path else None

    # Load vocabulary from specified directory, if available
    vocab_file = "vocab.json" if not pretrained_path else f"{pretrained_path}/vocab.json"
    if os.path.exists(vocab_file):
        vocab = GeneVocab.from_file(vocab_file)
        print(f"Loaded vocab from {vocab_file}")
    else:
        print("Vocab file not found! Ensure vocab.json is available.")
        return None

    # Load model configuration if a directory is provided
    if pretrained_path:
        config_file = f"{pretrained_path}/args.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                model_configs = json.load(f)
            print(f"Loaded model config from {config_file}")
        else:
            print("Model config not found! Using default hyperparameters.")
            model_configs = HYPERPARAMS  # Default values
    else:
        model_configs = HYPERPARAMS  # Default values

    # Initialize the model
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        vocab=vocab,
        dropout=model_configs["dropout"],
    )

    # Load pre-trained weights if available
    if model_file and os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")), strict=False)
        print(f"Loaded pre-trained scGPT model from {model_file}")
    else:
        print("No pre-trained model found. Training from scratch.")

    return model
