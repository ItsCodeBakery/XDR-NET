# XDR-Net: Explainable Hybrid CNN→Token-Attention for Diabetic Retinopathy Detection

**One-line:** A lightweight, explainable DR grader that couples an EfficientNet backbone with a single self-attention bridge, delivering strong performance and clear Grad-CAM evidence on APTOS 2019.

---

## 🧩 Overview

Automated diabetic retinopathy (DR) screening demands models that are **accurate, efficient, and transparent**. Pure CNNs capture local lesion cues (microaneurysms, hemorrhages, exudates) but can miss **global retinal context**; attention models capture global relations but are often heavy. **XDR-Net** balances both: it keeps EfficientNet’s compact convolutional features and adds a single token-level self-attention block before pooling. We pair this with a pragmatic training recipe for imbalanced grades and clinician-oriented visual explanations.

---

## 🧼 Preprocessing (APTOS 2019)

We apply a reproducible, screening-oriented preprocessing pipeline:

- Circular crop of the fundus and background removal  
- Resize to **384×384**, per-channel standardization  
- **CLAHE** on luminance to enhance local lesion contrast while limiting noise

**CLAHE example**

![CLAHE](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/ClahePrep.png)

---

## 🏗️ Proposed Methodology (XDR-Net)

**Backbone:** EfficientNet (from `timm`) for compact, high-quality feature extraction  
**Token-Attention bridge:** a **single Multi-Head Self-Attention (MHA)** applied on the final feature map (tokenized) to inject **global retinal context** with minimal overhead  
**Head:** Global Average Pooling → LayerNorm → Dropout → Linear (320→5) for 5-class DR grading  
**Training/Inference:** AdamW (cosine schedule), class-weighted CE with label smoothing; test-time augmentation and temperature scaling for calibrated probabilities  
**Explainability:** Grad-CAM on the last conv block; class-wise grids and error panels

**Architecture sketch**

![Methodology](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/METHODLOGY.png)

---

## 📈 Results (APTOS 2019)

**Validation split:** 1,805 images (five classes)

- **XDR-Net:** **Accuracy 97.40%**, **Macro-F1 97.38%**  
- **Baselines (same pipeline):**
  - ResNet-18 — Acc **81.82%**, Macro-F1 **68.39%**
  - ResNet-50 — Acc **82.73%**, Macro-F1 **70.32%**
  - ConvNeXt-Base — Acc **81.04%**, Macro-F1 **66.18%**

**Confusion matrix (counts + normalized)**

![Confusion Matrix](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/confusion_matrix_counts_vs_normalized.png)

---

## 🧐 Explainability

We generate class-discriminative **Grad-CAM** overlays from the final convolutional block, enabling graders to visualize lesion evidence and failure modes.

**Class-wise Grad-CAM montage**

![Grad-CAM](https://github.com/ItsCodeBakery/XDR-NET/blob/main/Proposed%20Methodology/plots/xdrGradCam.png)

---

## 📂 Repository Layout

