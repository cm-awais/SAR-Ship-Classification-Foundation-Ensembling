# Benchmarking Foundation Models for Imbalanced SAR Ship Classification

This repository contains a comprehensive benchmarking study of foundation models for SAR (Synthetic Aperture Radar) ship classification with class imbalance. The main work is presented in the `ensembling_foundation_models.ipynb` Jupyter/Colab notebook.

## Overview

The notebook evaluates and ensembles **7 state-of-the-art foundation models** on imbalanced SAR ship classification tasks, examining their performance across different class frequency groups and demonstrating ensemble methods to improve robustness.

### Key Features
- **Multi-model evaluation**: Benchmarks 7 foundation models including SAR-specific and general remote sensing models
- **Imbalance analysis**: Evaluates performance on frequent, medium, and rare ship classes
- **Ensemble methods**: Implements embedding concatenation and logits fusion strategies
- **Comprehensive metrics**: Reports accuracy, F1-macro, F1-weighted, precision, and recall per class

## Foundation Models

The notebook evaluates the following models:

1. **DOFA** - Domain-agnostic Vision Transformer for remote sensing
2. **SSL4EO-S12** - Self-Supervised Learning model for Earth Observation (Sentinel-2)
3. **ScaleMAE** - Large-scale SAR Masked Autoencoder
4. **Prithvi-100M** - IBM-NASA geospatial foundation model
5. **SAR-JEPA** - SAR domain-specific pretraining with Joint Embedding Predictive Architecture
6. **SARDet-100K (MSFA)** - Multiscale Fourier Angular features for SAR detection

## Datasets

Two SAR ship classification benchmarks are used:

### OpenSARShip
- **6 ship classes**: Cargo, Tanker, Dredging, Fishing, Passenger, Tug
- **Class groups**: Frequent (>500 samples), Medium (100-500), Rare (<100)
- Highly imbalanced dataset reflecting real-world scenarios

### FuSARShip
- **10 ship classes**: Cargo, Fishing, Bulk, Tanker, Container, Dredging, Tug, GeneralCargo, Passenger, and others
- Extended classification challenge with more fine-grained ship types

## Methodology

The notebook implements a **3-tier experimental framework**:

### Tier 1: Baseline (Individual Models)
- Extract embeddings from each foundation model independently
- Train linear classifiers on individual model embeddings
- Establish baseline performance for each model

### Tier 2: Imbalance Group Analysis
- Evaluate models separately on class frequency groups
- Analyze performance on frequent vs. medium vs. rare classes
- Identify model strengths across different data regimes

### Tier 3: Ensemble
- **Early Fusion**: Concatenate embeddings from all models
- **Logits Fusion**: Combine model predictions for meta-classifier training
- Demonstrates improved robustness through model combination

## Usage

### Running in Google Colab

1. Open the notebook in Google Colab:
   - Navigate to [Google Colab](https://colab.research.google.com/)
   - Upload `ensembling_foundation_models.ipynb` or open from GitHub

2. The notebook will automatically:
   - Install required dependencies (PyTorch, TorchGeo, TIMM, etc.)
   - Clone necessary repositories (SAR-JEPA, SARDet_100K)
   - Download and prepare datasets

3. Run cells sequentially to:
   - Load all 7 foundation models
   - Extract embeddings from SAR ship images
   - Train classifiers on individual and ensemble embeddings
   - Generate comprehensive evaluation metrics

### Local Execution

```bash
# Install Jupyter
pip install jupyter

# Install dependencies
pip install torch torchvision torchgeo==0.7.0 timm rasterio tifffile huggingface_hub einops numpy pillow matplotlib

# Launch notebook
jupyter notebook ensembling_foundation_models.ipynb
```

## Key Components

### SARDataset Class
- Custom PyTorch Dataset for loading SAR .tif images
- Handles normalization to [0, 1] range
- Manages class grouping by frequency
- Supports both OpenSARShip and FuSARShip formats

### Model Loading
- Automatic channel adaptation for SAR → RGB conversion
- Pretrained weights loading from Hugging Face Hub
- Standardized embedding extraction interface

### Evaluation Metrics
- Per-class accuracy, precision, recall
- Macro-averaged F1 score (equal weight to all classes)
- Weighted F1 score (weighted by class support)
- Confusion matrices for detailed error analysis

## Sample Results

**Tier 1 Performance (OpenSARShip - Individual Models)**:

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| SARDet-100K | 67.3% | 33.6% |
| DOFA | 66.7% | 39.4% |
| SSL4EO-S12 | 63.2% | 37.6% |
| SAR-JEPA | 56.6% | 29.2% |
| ScaleMAE | 47.4% | 24.1% |
| Prithvi-100M | 44.2% | 22.2% |

**Key Findings**:
- SAR-specialized models (SARDet-100K, SAR-JEPA) show strong performance on SAR-specific features
- General remote sensing models (DOFA, SSL4EO) demonstrate good generalization
- Ensemble methods improve robustness, particularly on imbalanced/rare classes
- Performance varies significantly across class frequency groups

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{sar-ship-classification-foundation-ensembling,
  title={Benchmarking Foundation Models for Imbalanced SAR Ship Classification},
  author={cm-awais},
  year={2024},
  howpublished={\url{https://github.com/cm-awais/SAR-Ship-Classification-Foundation-Ensembling}}
}
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- TorchGeo 0.7.0
- TIMM (PyTorch Image Models)
- Hugging Face Hub
- Rasterio, TiffFile
- NumPy, Pillow, Matplotlib

## License

Please refer to individual model licenses for usage restrictions. The notebook and evaluation framework are provided for research purposes.

## Acknowledgments

This work builds upon several foundation models:
- DOFA: Domain-agnostic Vision Transformer
- SSL4EO: Self-Supervised Learning for Earth Observation
- ScaleMAE: Large-scale SAR pretraining
- Prithvi: IBM-NASA geospatial model
- SAR-JEPA: SAR-specific pretraining
- SARDet-100K: SAR detection foundation

Special thanks to the authors of these models and the OpenSARShip/FuSARShip dataset curators.
