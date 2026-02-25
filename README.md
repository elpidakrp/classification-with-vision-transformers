<p align="center">
  <h1 align="center">Vision Transformers vs CLIP on Food-101</h1>
  <p align="center">
    Fine-Tuning vs Zero-Shot Learning | Attention Analysis | Model Behavior Comparison
  </p>
</p>

<p align="center">
  <b>University of Washington – EE562A – Advanced Deep Learning Project</b><br>
  Elpida Karapepera • Max Ramstad
</p>

---

# 🚀 Overview

This project presents a structured comparison between:

- Fine-tuned Vision Transformer (ViT) models  
- Zero-shot CLIP (Contrastive Language–Image Pretraining)

on the **Food-101 dataset (101 classes, 101k images)**.

We analyze:

- Accuracy performance
- Training efficiency
- Architectural differences
- Attention behavior
- Failure/success case asymmetries

The goal was not only to compare accuracy — but to understand *why* models behave differently.

---

# 🧠 Models Evaluated

## Fine-Tuned Vision Transformers

| Model | Architecture Type | Patch Strategy |
|-------|-------------------|---------------|
| `google/vit-base-patch16-224-in21k` | Standard ViT | 16×16 global patches |
| `microsoft/swinv2-tiny-patch4-window8-256` | Hierarchical Transformer | 4×4 patches, shifted windows |
| `vit-large-patch16-224-in21k` | Large-scale ViT | 16×16 global patches |

All models were initialized from ImageNet-21k pretraining and fine-tuned on Food-101.

---

## Zero-Shot Model

### CLIP (RN50)
- Pretrained multimodal model
- No fine-tuning performed
- Evaluated using text prompts for 101 food categories

Official repository:
https://github.com/openai/CLIP

---

# 📊 Results

## 🔹 Classification Accuracy

| Model | Training | Best Accuracy |
|--------|----------|---------------|
| **ViT Base** | 15 epochs | **84.8%** |
| **SwinV2 Tiny** | 24 epochs (early stop) | **84.5%** |
| CLIP (Zero-shot) | None | ~77% |
| ViT Large | 4 epochs | ~20% |

---

## 📈 Training Dynamics

<p align="center">
  <img src="assets/training_curves.png" width="750">
</p>

<p align="center">
  <em>Figure 1: Training and validation accuracy for ViT Base and SwinV2 during fine-tuning.</em>
</p>

**Observations:**

- Both models converge quickly due to ImageNet pretraining.
- ViT Base stabilizes slightly higher than SwinV2.
- Early stopping prevents overfitting in SwinV2.
- ViT-Large shows unstable early training and high compute cost.

Fine-tuning improves performance by ~7–8% over zero-shot CLIP.

---

# 🔍 Qualitative Analysis: Attention Behavior

Accuracy alone does not explain *why* models succeed or fail.  
We therefore performed cross-model attention analysis.

### Cross-Model Correct Predictions

- CLIP correct, ViTs wrong → **540 images**
- ViT Base correct, others wrong → **593 images**
- SwinV2 correct, others wrong → **1742 images**

---

## 🧠 Cross-Model Attention Comparison

<p align="center">
  <img src="assets/attention_grid.png" width="850">
</p>

<p align="center">
  <em>Figure 2: Aggregated attention maps for CLIP, ViT Base, and SwinV2 on the same input image.</em>
</p>

Architectural biases become visible:

- **CLIP** emphasizes global semantic structure.
- **ViT Base** balances object-level and global reasoning.
- **SwinV2** focuses strongly on localized texture due to windowed attention.

---

## 🔴 Representative Failure Case

**True Label:** hamburger  

| Model | Prediction |
|--------|------------|
| CLIP | hamburger ✅ |
| ViT Base | pulled_pork_sandwich |
| SwinV2 | beef_tartare |

<p align="center">
  <img src="assets/hamburger_case.png" width="750">
</p>

<p align="center">
  <em>Figure 3: SwinV2 attends primarily to meat texture, while CLIP captures full sandwich structure.</em>
</p>

This example highlights how hierarchical windowed attention
can overweight fine-grained texture, whereas CLIP leverages broader scene context.

---

## 🔹 Key Takeaways

- Fine-tuning significantly improves domain performance.
- Zero-shot CLIP remains competitive despite no task-specific training.
- Hierarchical transformers introduce strong local feature bias.
- Attention visualization reveals structural reasoning differences
  not captured by scalar accuracy metrics.

---

## 🔹 Key Insights

- Initial ViT accuracy (~78%) closely matched CLIP (~77%), confirming architectural similarity.
- Fine-tuning improved performance by ~7–8%.
- SwinV2 achieved comparable accuracy with improved efficiency.
- ViT-Large required excessive compute (~10 hours for 4 epochs) and was discontinued.

---

# 🔍 Attention Map Analysis

We performed a detailed cross-model error analysis.

### Cross-Model Correct Predictions

- CLIP correct, ViTs wrong → **540 images**
- ViT Base correct, others wrong → **593 images**
- SwinV2 correct, others wrong → **1742 images**

---

## 🧠 Behavioral Observations

### CLIP
- Strong global semantic reasoning
- Focuses on overall scene composition
- More robust to ambiguous local features

### ViT Base
- Balanced global and object-level focus
- Best overall performance

### SwinV2
- Strong local feature sensitivity
- Windowed attention emphasizes fine details

---

## Example Failure Case

**True Label:** hamburger  

| Model | Prediction |
|--------|------------|
| CLIP | hamburger ✅ |
| ViT Base | pulled_pork_sandwich |
| SwinV2 | beef_tartare |

SwinV2 focused primarily on meat texture due to windowed attention, while CLIP captured full sandwich structure.

---

# 🏗️ Technical Implementation

## Training

- Framework: PyTorch + HuggingFace Transformers
- Loss: Cross-Entropy
- Hardware: Google Colab Pro+ (GPU)
- Early stopping enabled
- RGB normalization (grayscale images removed)

## Evaluation

Metrics:
- Validation Accuracy
- Validation Loss

Validation metrics prioritized to avoid overfitting bias.

---

# 🎨 Attention Visualization Pipeline

To extract attention maps:

1. Hooked into attention layers
2. Pulled attention tensors
3. Reconstructed patch/window grids
4. Rescaled heatmaps
5. Overlayed attention maps onto images

Each model contains 12 attention heads.
For visualization clarity, heads were aggregated.

Resources adapted:
- Transformer-MM-Explainability
- Facebook DINO visualization tools
- CLIP Grad-CAM examples

---

# 🧪 Experimental Setup

- Dataset: Food-101 (101 classes)
- Environment: Colab Pro+
- Training:
  - ViT Base → 15 epochs
  - SwinV2 → 30 epochs (early stopped at 24)
  - ViT Large → discontinued

---

# ⚙️ Dependencies
torch
torchvision
transformers
datasets
evaluate
timm
einops
captum
scipy
scikit-image
matplotlib
pillow
ftfy
tensorboard
git+https://github.com/openai/CLIP.git


---

# 📎 Notebooks

Training & experiments conducted in Google Colab:

- ViT Training  
- SwinV2 Training  
- CLIP Evaluation  
- Attention Visualization  

(Links available in repository.)

---

# 🏁 Conclusions

- Fine-tuning remains critical for domain-specific performance.
- Zero-shot CLIP performs remarkably well given no task-specific training.
- Hierarchical transformers (SwinV2) offer strong performance with efficient computation.
- Attention analysis reveals structural bias differences between global and windowed attention mechanisms.

**Best Model:**  
`google/vit-base-patch16-224-in21k` — 84.8%

---

# 👩‍💻 Contributions

### Elpida Karapepera
- CLIP implementation
- SwinV2 implementation
- Cross-model failure analysis
- Attention visualization engineering

### Max Ramstad
- ViT Base implementation
- Dataset bug resolution (grayscale issue)
- CLIP attention visualization refinement

Joint:
- Experimental design
- Evaluation
- Report and presentation

---

# 📚 References

- OpenAI CLIP  
- HuggingFace Transformers  
- Transformer-MM-Explainability  
- Facebook Research DINO  

---

<p align="center">
  Built with PyTorch • Transformers • Attention Interpretability
</p>
