# Vision Transformers vs CLIP on Food-101  
**EE562A – University of Washington - Final Project**  
Elpida Karapepera, Max Ramstad  

---

## 📌 Project Overview

This project compares fine-tuned Vision Transformer (ViT) models with a zero-shot CLIP model on the Food-101 dataset.

The objectives of this project were:

- Train at least two Vision Transformer classifiers on Food-101
- Evaluate a pre-trained CLIP model in a zero-shot setting (no fine-tuning)
- Compare performance across models
- Identify failure/success cases between models
- Visualize and analyze attention maps
- Study architectural differences and their effect on classification behavior

---

## 🧠 Models Used

### 1️⃣ Fine-Tuned Vision Transformers

- `google/vit-base-patch16-224-in21k`
- `microsoft/swinv2-tiny-patch4-window8-256`
- `vit-large-patch16-224-in21k` (training discontinued)

All ViT models were initialized from ImageNet-21k pretraining and fine-tuned on Food-101.

### 2️⃣ CLIP (Zero-Shot Evaluation)

We used the RN50 variant from OpenAI’s official CLIP repository:

https://github.com/openai/CLIP

The model was evaluated without any additional training.

---

## 📊 Results Summary

| Model | Training | Best Accuracy |
|--------|----------|---------------|
| ViT Base | 15 epochs | **84.8%** |
| SwinV2 Tiny | 24 epochs (early stopped) | **84.5%** |
| CLIP (RN50) | Zero-shot | ~77% |
| ViT Large | 4 epochs | ~20% |

### Key Observations

- Initial ViT accuracy (~78%) closely matched CLIP (~77%), confirming architectural similarity.
- Fine-tuned ViTs significantly outperform zero-shot CLIP.
- SwinV2 performs competitively due to hierarchical attention and smaller patches.
- ViT Large required excessive compute (~10 hours for 4 epochs) and showed poor early performance.

---

## 🔍 Architectural Differences

### ViT Base
- 16×16 patch size
- Global self-attention
- Strong global context modeling

### SwinV2 Tiny
- 4×4 patches
- 8×8 shifted windows
- Hierarchical transformer structure
- More computationally efficient
- Better at capturing local details

In practice:
- CLIP emphasizes global semantics.
- ViT Base balances global and local context.
- SwinV2 focuses more on fine-grained details.

---

## 🎯 Attention Map Analysis

We compared predictions across models and extracted cases where:

- CLIP was correct and ViTs failed (540 images)
- ViT Base was correct and others failed (593 images)
- SwinV2 was correct and others failed (1742 images)

### Observed Patterns

- CLIP often captures overall scene semantics.
- ViT Base balances object-level and global attention.
- SwinV2 tends to focus strongly on local features due to windowed attention.

Attention maps were extracted by:
- Pulling attention tensors
- Reconstructing patch/window grids
- Rescaling heatmaps
- Overlaying them onto original images

Each model contains 12 attention heads. We aggregate them for visualization.

---

## 🛠️ Implementation Details

### Training Setup
- Framework: PyTorch + HuggingFace Transformers
- Loss: Cross-Entropy
- Hardware: Google Colab Pro+ GPU
- Early stopping applied
- Dataset preprocessing: ensured RGB-only images (removed grayscale images)

### Metrics
- Validation Accuracy
- Validation Loss

Validation metrics were prioritized to avoid overfitting.

---

## 📦 Dependencies
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

## 🚀 Colab Notebooks

### Training ViT (Main Notebook)
https://colab.research.google.com/drive/1NDA4f6dxc_I2vboEdjBWVG5WOw9nBWP0

### ViT Base & Large
https://colab.research.google.com/drive/1mu1z-SDgGbJFD1M7jLxVrAzsm6xu3Wta

### SwinV2 Training
https://colab.research.google.com/drive/1Hp7coeC0eGtW1WgfaPoEZuwhpR7rJ7j2

### Attention Visualization
https://colab.research.google.com/drive/1N4wOUsFzKzG_XUuMpszpDFa6mt7ZzO2W

---

## 🧪 Experimental Setup

- Dataset: Food-101
- Hardware: Colab Pro+ GPU
- Training:
  - ViT Base: 15 epochs
  - SwinV2 Tiny: 30 epochs (early stopped at 24)
  - ViT Large: 4 epochs (discontinued)

---

## 🏁 Conclusions

- Fine-tuning significantly improves performance over zero-shot CLIP.
- SwinV2 achieves strong performance with fewer parameters.
- Large ViT models are computationally expensive to train.
- Architectural differences directly influence attention behavior and error patterns.

The best performing model was:
**google/vit-base-patch16-224-in21k (84.8%)**

---

## 👩‍💻 Contributions

### Elpida Karapepera
- CLIP implementation
- SwinV2 implementation
- Model comparison logic
- Attention visualization (CLIP + ViTs)

### Max Ramstad
- ViT Base implementation
- Dataset bug fix (grayscale issue)
- CLIP attention visualization (v2)

### Both
- Experiments
- Analysis
- Report
- Presentation

---

## 📚 References

- OpenAI CLIP: https://github.com/openai/CLIP  
- Transformer-MM-Explainability  
- Facebook Research DINO attention visualization  

---

## 📎 License

This repository was developed for academic purposes (EE562A Final Project).
