# Articles Repository

This repository contains code implementations and experiments related to various technical articles I've written.

## Projects

### QLoRA Quantization
- **Location**: `QLoRA_quantization/`
- **Description**: Implementation of QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning of large language models
- **Key Features**:
  - 4-bit quantization with NF4 format
  - Memory-efficient fine-tuning using LoRA adapters
  - Training on Mistral-7B-Instruct model
- **Files**:
  - `qlora_training.ipynb` - Main training notebook
  - `inspect_model.ipynb` - Model inspection and analysis
  - `inspect_trained_model.py` - Python script for model inspection

### SVD Applications
- **Location**: `article_svd/`
- **Description**: Various applications of Singular Value Decomposition (SVD) in machine learning and data analysis
- **Key Features**:
  - Image compression using SVD
  - Latent Semantic Analysis (LSA)
  - Movie recommendation systems
  - PCA implementation using SVD
- **Files**:
  - `image_compression_svd.py` - Image compression demo
  - `latent_semantic_analysis.py` - LSA implementation
  - `movie_recommendations_svd.py` - Recommendation system
  - `pca_svd.py` - PCA using SVD
  - `LSA.png`, `PCA.png` - Visualization outputs

## Setup

### Environment Setup
```bash
# Create virtual environment (if not exists)
python -m venv venv312

# Activate virtual environment
# On Windows:
venv312\Scripts\activate
# On Unix/MacOS:
source venv312/bin/activate

# Install dependencies (requirements.txt should be added)
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter

## Usage

Each project is self-contained with its own notebooks and scripts. Navigate to the respective directory and follow the instructions within the code files.

### Running Jupyter Notebooks
```bash
jupyter notebook
# or
jupyter lab
```

## Repository Structure
```
articles/
├── QLoRA_quantization/          # QLoRA implementation
│   ├── qlora_training.ipynb     # Main training notebook
│   ├── inspect_model.ipynb       # Model inspection
│   ├── inspect_trained_model.py  # Model inspection script
│   └── qlora_results/            # Training outputs and checkpoints
├── article_svd/                  # SVD applications
│   ├── image_compression_svd.py
│   ├── latent_semantic_analysis.py
│   ├── movie_recommendations_svd.py
│   ├── pca_svd.py
│   ├── LSA.png
│   └── PCA.png
├── venv312/                      # Virtual environment
└── README.md                     # This file
```

## Contributing

This is a personal repository for article-related code. Feel free to explore the implementations and adapt them for your own use cases.

## License

This repository contains educational and experimental code. Please check individual files for specific licensing information.

## Contact

For questions about the implementations or related articles, please refer to the original article sources or open an issue in this repository.
