# XDR-Net: A Hybrid Convolutional Network with a Single-Block Attention Bridge and Balanced Optimization for Diabetic Retinopathy Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Backbone: EfficientNet-B0](https://img.shields.io/badge/Backbone-EfficientNet--B0-orange.svg)](https://github.com/huggingface/pytorch-image-models)

Official implementation and reproducibility materials for **XDR-Net**, a compact hybrid CNN-attention framework for five-class diabetic retinopathy (DR) grading from color fundus photographs.

XDR-Net combines retinal preprocessing, an EfficientNet-B0 convolutional backbone, a **single-block attention bridge** for global contextual interaction, effective-number-based class balancing, and post-hoc Grad-CAM visualization. The current manuscript evaluates the framework on APTOS 2019, EyePACS, Messidor, and IDRiD.

> **Current APTOS 2019 result:** 97.38% accuracy and 97.38% macro-F1 on the fixed image-level evaluation partition used in the manuscript.

---

## Overview

Diabetic retinopathy grading is challenging because public fundus datasets contain substantial class imbalance, acquisition variability, subtle retinal lesions, and ambiguity between adjacent severity grades. XDR-Net is designed to retain efficient convolutional lesion representation while introducing limited-cost global contextual modeling.

The current pipeline contains four main components:

1. **Retinal preprocessing** — circular retinal cropping, resizing to 384×384, per-channel normalization, CLAHE-based contrast enhancement, and mild training-time augmentation.
2. **EfficientNet-B0 backbone** — convolutional extraction of local retinal features.
3. **Single-block attention bridge** — global interaction among the final 12×12 spatial feature locations (144 tokens, token dimension 320, four attention heads).
4. **Balanced optimization** — effective-number-derived class weights used for class-aware sampling and class-weighted, label-smoothed cross-entropy.

Grad-CAM is generated from the final convolutional block for post-hoc visualization. For IDRiD, pixel-level lesion masks are **not used for model training**; they are used only for quantitative post-hoc evaluation of Grad-CAM localization.

---

## Performance Summary

| Dataset | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) | Macro-AUC (%) |
|---|---:|---:|---:|---:|---:|
| **APTOS 2019** | **97.38** | **97.38** | **97.43** | **97.40** | **97.99** |
| **EyePACS** | **97.45** | **98.35** | **98.90** | **97.80** | **98.90** |
| **Messidor** | **96.82** | **96.95** | **97.50** | **96.40** | **97.50** |
| **IDRiD** | **95.90** | **96.17** | **96.75** | **95.60** | **96.75** |

These values correspond to the current manuscript tables and should be interpreted within the reported benchmark protocols. They do not constitute prospective or independent external clinical validation.

---

## Dataset Partitioning

All partitions are created **before** augmentation or class rebalancing and are kept fixed across XDR-Net and the reproduced baseline models.

Reliable patient identifiers are not consistently available across the four public datasets. Therefore, the manuscript uses **fixed image-level partitions**, and patient-level independence cannot be verified.

| Dataset | Total images | Training | Validation | Test | Seed | Partition level |
|---|---:|---:|---:|---:|---:|---|
| APTOS 2019 | 3,662 | 3,112 | 550 | N/A | 42 | Image level |
| EyePACS | 35,126 | 28,102 | 3,512 | 3,512 | 42 | Image level |
| Messidor | 1,200 | 960 | 120 | 120 | 42 | Image level |
| IDRiD | 516 | 330 | 83 | 103 | 42 | Image level |

For APTOS 2019, the public competition test labels are unavailable, so the fixed validation subset is used as the final evaluation cohort. For IDRiD, the separate test subset is retained for evaluation.

---

## Architecture

### EfficientNet-B0 feature extraction

For a 384×384 input image, the final EfficientNet-B0 convolutional feature tensor has spatial resolution 12×12 and 320 channels:

```text
Input:              3 × 384 × 384
Final feature map:  320 × 12 × 12
Spatial tokens:     144
Token dimension:    320
```

### Single-block attention bridge

The final convolutional feature map is reshaped into 144 spatial tokens and processed by a lightweight four-head multi-head self-attention bridge. The attention operation allows spatially distant retinal regions to exchange contextual information while operating at the low-resolution final feature stage.

```text
Token count (Nt):     144
Token dimension (d):  320
Attention heads (h):  4
Per-head dimension:   80
```

The current manuscript reports **4.219 M total trainable parameters** for XDR-Net. Computational figures reported below follow the same single-image profiling protocol used in the revised manuscript.

---

## Preprocessing and Augmentation

### Deterministic preprocessing

1. Retinal-region/circular cropping to suppress non-retinal borders.
2. Resize to 384×384 pixels.
3. Per-channel normalization.
4. CLAHE-based contrast enhancement on the luminance channel.

### Training-time augmentation

Mild geometric and photometric transforms are applied only during training, including horizontal flipping, rotation/resize-crop, slight color jitter, and Gaussian blur.

Validation and test preprocessing remain deterministic. Optional test-time augmentation (TTA) averages predictions over **M = 4** geometric views.

---

## Class-Imbalance Handling

Let `n_c` denote the number of training examples in class `c`. XDR-Net derives class weights from the effective number of samples and normalizes the weights to unit mean.

The same class-weight information is used in two places:

- **Class-aware sampling**, increasing exposure to minority DR grades.
- **Class-weighted, label-smoothed cross-entropy**, increasing the contribution of minority-class examples to optimization.

Current effective-number settings:

```text
APTOS 2019: beta ≈ 0.999
EyePACS:    beta ≈ 0.9999
Messidor:   beta ≈ 0.995
IDRiD:      beta ≈ 0.995
```

Label smoothing is set to **epsilon = 0.1** for the main experiments. This README does **not** claim a quantitative epsilon sensitivity sweep unless the corresponding experimental results are released and reported in the manuscript.

---

## Training Configuration

```yaml
Input:
  image_size: 384x384
  batch_size: 32

Optimization:
  optimizer: AdamW
  beta1: 0.9
  beta2: 0.999
  initial_learning_rate: 3e-4
  weight_decay: 1e-4
  learning_rate_schedule: cosine decay
  gradient_clipping_l2: 1.0
  maximum_epochs: 20
  early_stopping_patience: 3
  early_stopping_metric: validation macro-F1
  checkpoint_selection: highest validation macro-F1

Regularization:
  label_smoothing: 0.1
  mixed_precision: enabled

Inference:
  tta_views: 4
```

Early stopping is triggered when validation macro-F1 fails to improve for **three consecutive epochs**. The checkpoint with the highest validation macro-F1 is retained for final evaluation.

---

## Reproducibility and Random Seeds

The main experimental protocol uses random seed **42** for dataset partitioning and reproducible training initialization. The implementation seeds Python, NumPy, PyTorch, and available CUDA devices, and deterministic cuDNN execution is enabled where required.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The manuscript additionally reports a five-run analysis for stochastic-training variability. Mean and standard deviation values should be interpreted exactly as reported in the corresponding revised manuscript table; this README does not summarize them with a single universal standard-deviation bound.

---

## Experimental Environment

The revised manuscript reports the following environment for the main experiments:

```text
GPU:            NVIDIA Tesla T4, 16 GB
Python:         3.12.13
PyTorch:        2.10.0
CUDA:           12.8
cuDNN:          9.10.2
torchvision:    0.25.0
timm:           1.0.26
Albumentations: 2.0.8
OpenCV:         4.13.0
scikit-learn:   1.6.1
NumPy:          2.0.2
pandas:         2.3.3
```

Automatic mixed precision (AMP) and channels-last memory format are enabled during training.

---

## Computational Efficiency

All computational metrics in the revised manuscript are profiled using a single NVIDIA Tesla T4 GPU at 384×384 input resolution and batch size `B = 1`, using warm-up runs followed by repeated timing iterations.

For the APTOS ablation configuration:

| Configuration | Params (M) | GFLOPs | Latency (ms/image) | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|---:|
| V4: XDR-Net, single pass | 4.219 | 0.83 | 9.86 | 94.95 | 95.12 |
| V5: XDR-Net + TTA (`M=4`) | 4.219 | 3.32 | 39.44 | 97.38 | 97.38 |

For V5, GFLOPs represent the total compute across all four augmentation passes. TTA therefore improves the reported final prediction metrics at the cost of approximately fourfold inference compute relative to a single pass.

---

## Grad-CAM Explainability

Grad-CAM is computed from the final convolutional block before the attention bridge. The resulting saliency maps are post-hoc visualizations of image regions associated with the model prediction.

### Quantitative IDRiD localization analysis

IDRiD pixel-level lesion annotations are **not used as training supervision**. They are used only for post-hoc evaluation of Grad-CAM localization on correctly classified images containing at least one annotated lesion.

Current manuscript results:

| Metric | Result |
|---|---:|
| Evaluated images | 82 |
| Pointing-game accuracy | 84.15% |
| Mean lesion-region energy ratio | 58.74% |
| Mean threshold-based IoU | 0.36 |

These metrics quantify spatial agreement between Grad-CAM activation and annotated lesion regions. They should not be interpreted as proof of causal model reasoning or as clinically validated diagnostic explanations.

---

## Ablation Summary

The current manuscript evaluates the following incremental configurations across all four datasets:

- **V1:** EfficientNet-B0 backbone only
- **V2:** + CLAHE preprocessing
- **V3:** + class reweighting
- **V4:** + single-block attention bridge
- **V5:** + test-time augmentation (final XDR-Net configuration)

APTOS 2019 results:

| Variant | Val. Acc. (%) | Macro-F1 (%) | Macro-AUC (%) | Params (M) | GFLOPs | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| V1: Backbone only | 80.90 | 63.94 | 91.87 | 4.014 | 0.79 | 11.20 |
| V2: + CLAHE | 80.08 | 64.55 | 89.98 | 4.014 | 0.79 | 10.94 |
| V3: + Class reweighting | 78.04 | 59.71 | 90.49 | 4.014 | 0.79 | 10.78 |
| V4: + Attention bridge | 94.95 | 95.12 | 95.60 | 4.219 | 0.83 | 9.86 |
| **V5: + TTA (XDR-Net)** | **97.38** | **97.38** | **97.99** | **4.219** | **3.32** | **39.44** |

---

## Repository Structure

```text
XDR-NET/
├── Proposed Methodology/
│   ├── code/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── utils.py
│   │   └── gradcam.py
│   └── plots/
├── BaseLineExperement/
├── split_files/
├── requirements.txt
├── README.md
└── LICENSE
```

Exact filenames may vary as the repository is synchronized with the revised manuscript. Dataset split manifests and relevant reproducibility materials should be retained with the released implementation.

---

## Installation

```bash
git clone https://github.com/ItsCodeBakery/XDR-NET.git
cd XDR-NET
pip install -r requirements.txt
```

Datasets are not redistributed in this repository. Please obtain APTOS 2019, EyePACS, Messidor, and IDRiD from their respective official providers and follow the corresponding access and usage conditions.

---

## Training and Evaluation

Example commands depend on the released scripts and argument definitions. The training configuration used for the manuscript should correspond to the settings documented above: 384×384 input, batch size 32, AdamW, initial learning rate `3e-4`, maximum 20 epochs, early-stopping patience 3, and checkpoint selection by validation macro-F1.

Please use the released fixed split manifests rather than generating new partitions when reproducing the manuscript results.

---

## Scope and Limitations

The reported experiments use public benchmark datasets and do not constitute independent prospective clinical validation. Because reliable patient identifiers are not consistently available, patient-level separation cannot be verified. Generalization to independent institutions, out-of-distribution data, different imaging devices, and severely degraded real-world fundus images remains to be established.

Accordingly, XDR-Net should be regarded as a **promising benchmark research framework requiring further external and prospective validation**, rather than as a deployment-ready clinical system.

---

## Data Availability

The study uses publicly available datasets:

- **APTOS 2019 Blindness Detection** — Kaggle
- **EyePACS Diabetic Retinopathy Detection** — Kaggle
- **Messidor** — ADCIS
- **IDRiD** — IEEE DataPort

The datasets are accessed and used according to the access and usage conditions stated by their respective providers.

---

## License

This repository is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---



---

## Contact

For questions about the implementation or reproducibility materials, please use the repository issue tracker or the correspondence information provided in the manuscript.

---

