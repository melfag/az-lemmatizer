# Azerbaijani Context-Aware Lemmatizer

Implementation of "Building a Context-Aware Lemmatizer for Azerbaijani Using BiLSTM and BERT: A Neural Approach to Morphological Analysis"

## Overview

This project implements a neural lemmatization system for Azerbaijani that combines:
- **Character-level BiLSTM** for morphological pattern recognition
- **AllmaBERT** (Azerbaijani BERT) for contextual understanding
- **Attention mechanism** for precise character-level decoding
- **Specialized components** for Azerbaijani linguistic phenomena (vowel harmony, consonant alternations)

## Architecture

```
Input Word + Context → [Character BiLSTM] → Character Representation
                       [AllmaBERT]        → Contextual Representation
                                ↓
                         [Gated Fusion]
                                ↓
                    [LSTM Decoder with Attention]
                                ↓
                           Lemma Output
```

## Setup

### Prerequisites
- Python 3.9+
- Apple Silicon (M1/M2/M4) or CUDA-capable GPU
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd azerbaijani-lemmatizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

This project uses the DOLLMA dataset (651M words):
- Source: https://huggingface.co/datasets/allmalab/DOLLMA
- Automatically downloaded during first run

## Models

Supports multiple AllmaBERT variants:
- `allmalab/bert-small-aze` (46M parameters)
- `allmalab/bert-base-aze` (135M parameters) - **Default**
- `allmalab/bert-large-aze` (370M parameters)

## Usage

### Training

```bash
# 1. Prepare data
python scripts/prepare_data.py --dataset allmalab/DOLLMA --output data/processed

# 2. Create specialized test sets (optional but recommended)
python scripts/create_specialized_tests.py \
    --test-file data/processed/test.json \
    --output-dir data/specialized_tests

# 3. Train model
python scripts/train.py --config configs/training_config.yaml

# Train with specific BERT variant
python scripts/train.py --bert-model allmalab/bert-small-aze
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --model-path checkpoints/best_model.pt

# Run error analysis
python scripts/evaluate.py --model-path checkpoints/best_model.pt --error-analysis
```

### Inference

```bash
# Interactive mode
python scripts/inference.py --model-path checkpoints/best_model.pt

# Batch inference
python scripts/inference.py --model-path checkpoints/best_model.pt --input input.txt --output output.txt
```

## Project Structure

```
azerbaijani-lemmatizer/
├── data/                    # Data files
├── models/                  # Model implementations
├── utils/                   # Utilities
├── training/                # Training logic
├── evaluation/              # Evaluation framework
├── configs/                 # Configuration files
├── scripts/                 # Executable scripts
└── notebooks/               # Jupyter notebooks
```

## Configuration

Edit `configs/model_config.yaml` and `configs/training_config.yaml` to customize:
- Model architecture (hidden sizes, layers)
- Training hyperparameters (learning rate, batch size)
- Data paths and preprocessing options

## Hardware Optimization

### Apple Silicon (M1/M2/M4)
- Automatically uses MPS (Metal Performance Shaders)
- Mixed precision training (FP16) enabled by default
- Optimized batch sizes for unified memory

### CUDA GPUs
- Automatically detected and used
- Supports distributed training (coming soon)

## Performance

Expected performance on test set:
- **Accuracy**: ~94.8%
- **Edit Distance**: ~0.11
- **F1 Score**: ~98.1%

Performance on specialized test sets:
- **Ambiguity Resolution**: ~89.5%
- **Vowel Harmony**: ~96.3%
- **Consonant Alternation**: ~91.8%

## Citation

If you use this code, please cite the original thesis:

```bibtex
@mastersthesis{mammadaliyev2025contextaware,
  author = {Elfag Mammadaliyev},
  title = {Building a Context-Aware Lemmatizer for Azerbaijani Using BiLSTM and BERT: A Neural Approach to Morphological Analysis},
  school = {Khazar University},
  year = {2025}
}
```


## Acknowledgments

- DOLLMA dataset: aLLMA Lab
- AllmaBERT models: aLLMA Lab
- Based on research at Khazar University
