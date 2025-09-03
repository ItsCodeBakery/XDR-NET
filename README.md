# XDR-Net: Explainable Hybrid CNN→Token-Attention for Diabetic Retinopathy Detection

 A lightweight, explainable DR grader that marries an EfficientNet backbone with a single self-attention bridge, delivering strong performance and clear Grad-CAM evidence on APTOS 2019.

---

## 🧩 Overview

Diabetic retinopathy (DR) screening at scale needs models that are **accurate, efficient, and explainable**. Pure CNNs excel at local lesion features, while attention helps with global retinal context—but many attention models are heavy. **XDR-Net** strikes a practical balance:

- **Backbone:** EfficientNet (from `timm`) as a compact, high-quality feature extractor  
- **Token-Attention bridge:** a **single** multi-head self-attention (MHA) block applied on the final feature map (tokenized) to capture long-range retinal context with minimal overhead  
- **Head:** global average pooling → LayerNorm → Dropout → Linear(320→5) for 5-class DR grading  
- **Explainability:** Grad-CAM over the last conv block to visualize class-discriminative regions  

All experiments are run on **APTOS 2019** (Kaggle). Preprocessing includes circular crop, resize to 384×384, channel-wise standardization, and CLAHE for contrast enhancement.

---

## 🔬 Key Findings (APTOS 2019)

- **XDR-Net:** **Accuracy 97.40%**, **Macro-F1 97.38%** (validation, 1,805 images)
- **Baselines (same pipeline):**  
  - **ResNet-18:** Acc 81.82%, Macro-F1 68.39%  
  - **ResNet-50:** Acc 82.73%, Macro-F1 70.32%  
  - **ConvNeXt-Base:** Acc 81.04%, Macro-F1 66.18%

XDR-Net outperforms strong baselines while remaining light enough for practical deployment and offering case-level visual explanations.

---

## 📊 Visual Highlights

> These images are stored under `Proposed Methodology/plots` (and related subfolders).  
> If a thumbnail doesn’t render, verify the file exists at the listed path.

**Training Curves**
  
![Accuracy Curve](Proposed%20Methodology/plots/accuracy_curve.png)
![F1 Curve](Proposed%20Methodology/plots/f1_curve.png)

**Confusion Matrix (Validation)**

![Confusion Matrix](Proposed%20Methodology/plots/confusion_matrix_diverging_combined.png)

**Class-wise Grad-CAM Grid**

![Grad-CAM Classwise Grid](Proposed%20Methodology/gradcam_classwise_grid.png)

**Class Distribution (Before Balancing)**

![Class Distribution](Proposed%20Methodology/original_class_distribution.png)

---

## 🧠 Methodology (Brief)

1. **Preprocessing:** circular crop of the fundus, resize to 384×384, per-channel standardization, and (optionally) CLAHE to enhance local lesion contrast.  
2. **Backbone:** EfficientNet (timm) to extract spatial features with strong local inductive bias.  
3. **Token-Attention bridge:** last feature map is tokenized (H×W patches); a **single MHA block** models global dependencies (optic disc ↔ fovea context, diffuse lesion patterns) with low latency.  
4. **Head:** global average pooling → LayerNorm → Dropout → Linear, producing 5-class logits.  
5. **Training/Inference:** AdamW (cosine schedule), class-weighted CE + label smoothing, TTA at inference, and temperature scaling for calibrated probabilities.  
6. **Explainability:** Grad-CAM overlays from the last conv block; class-wise grids and error panels for qualitative assessment.

---

## 📂 Repository Layout

