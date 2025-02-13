import os

def install_dependencies():
    packages = [
        "torch==2.0.1", "torchtext==0.15.2",
        "scanpy", "scvi-tools", "wandb", "anndata", "leidenalg"
    ]
    
    os.system("pip uninstall -y torch torchvision torchaudio torchtext scgpt")
    os.system(f"pip install {' '.join(packages)}")
    if not os.path.exists("scGPT-main"):
        os.system("git clone https://github.com/bowang-lab/scGPT.git scGPT-main")
    os.system("pip install -e scGPT-main")


if __name__ == "__main__":
    install_dependencies()