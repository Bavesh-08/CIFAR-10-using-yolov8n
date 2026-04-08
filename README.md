<div align="center">

# 🚀 YOLOv8 CIFAR-10 Image Classifier

<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/CIFAR--10-10 Classes-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/Top--1 Accuracy-75.3%25-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Top--5 Accuracy-98.5%25-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Colab-Run%20Now-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />

<br/>

> **Classifying 10 object categories from CIFAR-10 using YOLOv8n-cls — comparing modern YOLO classification against traditional CNN approaches (ResNet).**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [YOLOv8 vs ResNet — Comparison](#-yolov8-vs-resnet--head-to-head-comparison)
- [Class Predictions](#-class-predictions)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## 🧠 Overview

This project fine-tunes **YOLOv8n-cls** (nano classification variant) on the **CIFAR-10** dataset — a benchmark dataset of 60,000 tiny 32×32 images spanning 10 classes. The goal is to evaluate how well a modern YOLO-based classifier performs on this well-known benchmark and compare it against previous CNN-based approaches like ResNet.

---

## 📦 Dataset

| Property | Value |
|---|---|
| Dataset | CIFAR-10 |
| Total Images | 60,000 |
| Training Split | 48,000 images |
| Validation Split | 5,000 images |
| Image Size | 32 × 32 pixels |
| Classes | 10 |
| Format | PNG (saved via torchvision) |

### 🏷️ Class Labels

```
0: airplane   1: automobile   2: bird    3: cat    4: deer
5: dog        6: frog         7: horse   8: ship   9: truck
```

---

## 🏗️ Model Architecture

**YOLOv8n-cls** (nano classification head) — pretrained on ImageNet-1k and fine-tuned on CIFAR-10.

```
Layer Stack:
  Conv (3 → 16, 3×3, s=2)        →   464   params
  Conv (16 → 32, 3×3, s=2)       →   4,672  params
  C2f (32 → 32, 1 block)         →   7,360  params
  Conv (32 → 64, 3×3, s=2)       →   18,560 params
  C2f (64 → 64, 2 blocks)        →   49,664 params
  Conv (64 → 128, 3×3, s=2)      →   73,984 params
  C2f (128 → 128, 2 blocks)      →   197,632 params
  Conv (128 → 256, 3×3, s=2)     →   295,424 params
  C2f (256 → 256, 1 block)       →   460,288 params
  Classify (256 → 10)            →   343,050 params
  ─────────────────────────────────────────────────
  Total: 1,451,098 parameters | 3.4 GFLOPs | 56 layers
```

> **Pretrained weights** transferred: 156 / 158 layers from ImageNet checkpoint.

---

## ⚙️ Training Configuration

```python
model    = YOLO("yolov8n-cls.pt")   # Pretrained nano classification model
epochs   = 5
imgsz    = 32                        # Native CIFAR-10 resolution
batch    = 64
workers  = 2
optimizer = AdamW (lr=0.000714, momentum=0.9)   # Auto-selected
amp      = True                      # Automatic Mixed Precision
device   = CUDA (Tesla T4 — 14913 MiB)
```

**Data Augmentation applied:**
- Random horizontal flip (`fliplr=0.5`)
- HSV color jitter (`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`)
- Random erasing (`erasing=0.4`)
- RandAugment auto-augmentation policy

---

## 📊 Results

### Training Progress

| Epoch | Loss | Top-1 Acc | Top-5 Acc |
|:---:|:---:|:---:|:---:|
| 1/5 | 2.041 | 52.9% | 93.7% |
| 2/5 | 1.405 | 64.4% | 96.6% |
| 3/5 | 1.218 | 69.9% | 97.6% |
| 4/5 | 1.120 | 72.2% | 98.0% |
| **5/5** | **1.048** | **75.2%** | **98.5%** |

### Final Validation Metrics

```
┌────────────────────────────────────────────┐
│   Top-1 Accuracy  :  75.28%                │
│   Top-5 Accuracy  :  98.46%                │
│   Fitness Score   :  0.8687                │
│   Inference Speed :  0.66 ms / image       │
│   Training Time   :  0.105 hours (~6 min)  │
└────────────────────────────────────────────┘
```

---

## ⚔️ YOLOv8 vs ResNet — Head-to-Head Comparison

> Comparing this YOLOv8n-cls model against our previous **ResNet-based CNN** trained on CIFAR-10.

| Metric | 🟦 YOLOv8n-cls (This Project) | 🟥 ResNet CNN (Previous Project) |
|---|:---:|:---:|
| **Top-1 Accuracy** | **75.3%** | ~68% |
| **Top-5 Accuracy** | **98.5%** | ~95% |
| **Parameters** | 1.45M | ~11M (ResNet-18 scale) |
| **Inference Speed** | **0.66 ms/img** | ~3–5 ms/img |
| **Training Time (5 epochs)** | **~6 min** | ~15–20 min |
| **Pretrained Weights** | ✅ ImageNet | ✅ / ❌ (optional) |
| **Framework** | Ultralytics / PyTorch | TensorFlow / Keras |
| **Input Size** | 32×32 | 32×32 |
| **Augmentation** | RandAugment + HSV | Basic (flip, crop) |
| **Optimizer** | AdamW (auto-tuned) | Adam |

### 🔍 Key Takeaways

- ✅ **YOLOv8n-cls outperforms ResNet CNN** despite having ~8× fewer parameters
- ✅ **Faster training and inference** — YOLO's C2f blocks are optimized for speed
- ✅ **Better generalization** — RandAugment and erasing regularization reduce overfitting
- ⚠️ **Both models** were only trained for **5 epochs** — accuracy would improve significantly with longer training (20–30 epochs)
- 📌 **ResNet** is still a strong baseline for transfer learning on larger/higher-res datasets

---

## 🔮 Class Predictions

Sample predictions on CIFAR-10 (after 5 epochs):

```python
model.names
# {0: 'airplane', 1: 'automobile', 2: 'bird',  3: 'cat',  4: 'deer',
#  5: 'dog',      6: 'frog',       7: 'horse', 8: 'ship', 9: 'truck'}
```

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install ultralytics torchvision pillow pyyaml
```

### 2. Prepare Dataset
```python
from torchvision import datasets
from pathlib import Path

# Downloads CIFAR-10 and organizes into YOLOv8 folder structure
train_ds = datasets.CIFAR10(root="./data", train=True, download=True)
# Save images into cifar10_yolo/train/<classname>/ and val/<classname>/
```

### 3. Train the Model
```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")
results = model.train(
    data="cifar10_yolo/",
    epochs=5,
    imgsz=32,
    batch=64,
    workers=2
)
```

### 4. Validate
```python
metrics = model.val(data="cifar10_yolo/cifar10-cls.yaml", imgsz=32)
print(f"Top-1: {metrics.top1:.4f} | Top-5: {metrics.top5:.4f}")
```

### 5. Predict on New Images
```python
results = model.predict("your_image.jpg")
print(results[0].probs.top1)   # Predicted class index
```

---

## 📁 Project Structure

```
yolov8-cifar10/
│
├── data/                        # CIFAR-10 raw download
├── cifar10_yolo/
│   ├── train/                   # 48,000 training images (per-class folders)
│   │   ├── airplane/
│   │   ├── automobile/
│   │   └── ...
│   ├── val/                     # 5,000 validation images
│   └── cifar10-cls.yaml         # Dataset config for Ultralytics
│
├── runs/classify/train5/
│   ├── weights/
│   │   ├── best.pt              # Best checkpoint
│   │   └── last.pt              # Last epoch checkpoint
│   └── results.csv              # Training metrics log
│
└── notebook.ipynb               # Full training notebook (Colab)
```

---

## 🧪 Future Improvements

- [ ] Train for 20–30 epochs to reach ~85%+ accuracy
- [ ] Try `yolov8s-cls.pt` (small) for better accuracy vs speed tradeoff
- [ ] Use the full 50k training set with no subset
- [ ] Add test-time augmentation (TTA) for better val scores
- [ ] Export to ONNX / TensorRT for deployment

---

## 👨‍💻 Author

**Bavesh V** — ML & AI Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-blue?style=flat-square)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

</div>

---

<div align="center">
  <sub>Part of an ongoing series of ML benchmark experiments — MNIST → CIFAR-10 (ResNet) → CIFAR-10 (YOLOv8)</sub>
</div>
