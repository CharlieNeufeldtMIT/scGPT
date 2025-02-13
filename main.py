from setup_environment import install_dependencies
from data_preprocessing import preprocess_data
from train import train
from evaluate import evaluate

def main():
    """Runs the full pipeline."""
    install_dependencies()
    preprocess_data("data/combined_expression_matrix.csv", "data/filtered_adata.h5ad")
    train()
    evaluate()

if __name__ == "__main__":
    main()
