# Vision Transformer (ViT) from Scratch

A clean, minimal implementation of the Vision Transformer (ViT) architecture in PyTorch, trained on MNIST for digit classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This repository contains a from-scratch implementation of the **Vision Transformer (ViT)** architecture, originally introduced in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

The implementation demonstrates core ViT concepts on the MNIST dataset, achieving **~98% accuracy** in just 5 epochs.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Vision Transformer (ViT)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input Image (28×28)                                                       │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────┐                              │
│   │         Patch Embedding                  │                              │
│   │   Split into 7×7 patches → 16 patches   │                              │
│   │   Linear projection → 64-dim vectors    │                              │
│   └─────────────────────────────────────────┘                              │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────┐                              │
│   │   [CLS] + Patch Tokens + Position Emb   │                              │
│   │        (1 + 16) × 64 dimensions         │                              │
│   └─────────────────────────────────────────┘                              │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────┐                              │
│   │      Transformer Encoder (×4)           │                              │
│   │   ┌───────────────────────────────┐     │                              │
│   │   │  Layer Norm                   │     │                              │
│   │   │  Multi-Head Attention (4 heads)│    │                              │
│   │   │  Residual Connection          │     │                              │
│   │   │  Layer Norm                   │     │                              │
│   │   │  MLP (64 → 128 → 64)          │     │                              │
│   │   │  Residual Connection          │     │                              │
│   │   └───────────────────────────────┘     │                              │
│   └─────────────────────────────────────────┘                              │
│         │                                                                   │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────┐                              │
│   │         MLP Head                        │                              │
│   │   Extract [CLS] token → LayerNorm       │                              │
│   │   Linear → 10 classes                   │                              │
│   └─────────────────────────────────────────┘                              │
│         │                                                                   │
│         ▼                                                                   │
│      Output: Class Probabilities (10)                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Key Components

| Component | Description |
|-----------|-------------|
| **Patch Embedding** | Splits 28×28 image into 16 patches (7×7 each) using Conv2d |
| **Class Token** | Learnable [CLS] token prepended to patch sequence |
| **Position Embedding** | Learnable positional encodings for all 17 tokens |
| **Transformer Encoder** | 4 stacked blocks with Multi-Head Self-Attention + MLP |
| **MLP Head** | Final classification layer operating on [CLS] token |

## 🔧 Hyperparameters

```python
num_classes = 10          # MNIST digits (0-9)
batch_size = 64
patch_size = 7            # 28/7 = 4 patches per dimension
num_patches = 16          # 4×4 grid of patches
embedding_dim = 64        # Transformer hidden dimension
attention_heads = 4       # Multi-head attention heads
transformer_blocks = 4    # Number of encoder layers
mlp_hidden_nodes = 128    # MLP intermediate dimension
learning_rate = 0.001     # Adam optimizer
epochs = 5
```

## 📈 Results

Training on MNIST dataset with the above configuration:

| Epoch | Training Loss | Training Accuracy |
|-------|--------------|-------------------|
| 1 | 323.42 | 88.93% |
| 2 | 115.07 | 96.23% |
| 3 | 82.18 | 97.29% |
| 4 | 67.76 | 97.73% |
| 5 | ~55 | ~98% |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision
```

### Run Training

```bash
# Clone the repository
git clone https://github.com/yourusername/ViT-from-scratch.git
cd ViT-from-scratch

# Run the notebook or Python script
jupyter notebook ViT.ipynb
```

## 📁 Project Structure

```
ViT-from-scratch/
├── README.md           # This file
├── ViT.ipynb          # Main implementation notebook
├── images/            # Architecture diagrams and visualizations
│   └── vit_architecture.png
└── requirements.txt   # Dependencies
```

## 🧠 Understanding the Code

### 1. Patch Embedding
```python
class PatchEmbedding(nn.Module):
    # Converts image into sequence of patch embeddings
    # Input: (B, 1, 28, 28) → Output: (B, 16, 64)
```

### 2. Transformer Encoder
```python
class TransformerEncoder(nn.Module):
    # Pre-norm architecture with:
    # - Multi-Head Self-Attention
    # - Feed-Forward MLP with GELU activation
    # - Residual connections
```

### 3. Vision Transformer
```python
class VisionTransformer(nn.Module):
    # Complete ViT pipeline:
    # Patch Embed → Add [CLS] → Add Position → Transformers → MLP Head
```

## 📚 References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Original ViT Paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer Architecture

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share feedback and suggestions

---

⭐ If you found this helpful, please consider giving the repository a star!
