# CNN-GAN Project: CNN Module

**Developer:** Snehangshu Pal  
**Institution:** Swami Vivekananda Institute of Science and Technology

## 📌 Project Overview

This module contains a ResNet-18 implementation trained on the CIFAR-10 dataset. It serves as both a standalone classifier and a feature extractor for the GAN module to calculate Perceptual Loss.

## 🏗️ Architecture

- **Model:** ResNet-18 (modified for CIFAR-10)
- **Dataset:** CIFAR-10 (10 classes, 32x32 RGB images)
- **Input:** 3x32x32 tensor
- **Output:** 10-dimensional logits

## 📂 Directory Structure

```
CNN/
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── train_cnn.py               # Main training script
├── accuracy_graph_plot_resnet.png  # Training performance visualization
├── models/
│   ├── __init__.py
│   └── resnet.py              # ResNet-18 architecture implementation
├── utils/
│   ├── __init__.py
│   ├── datasets.py           # CIFAR-10 DataLoaders & preprocessing
│   └── feature_extractor.py  # Feature extraction utilities for GAN
├── cnn/
│   ├── checkpoints/          # Model weights (created during training)
│   └── data/                # Dataset storage (created during training)
└── CNN_GAN/                  # GAN module (if present)
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib
```

### Training the Model

```bash
cd CNN
python train_cnn.py
```

**Training Configuration:**

- **Epochs:** 10
- **Batch Size:** 128
- **Optimizer:** SGD with Momentum (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **Learning Rate Schedule:** Decay at epochs 20 and 40
- **Device:** Automatic CUDA/CPU detection

### Using the Trained Model

```python
import torch
from models.resnet import ResNet18_cifar10

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18_cifar10().to(device)
model.load_state_dict(torch.load('cnn/checkpoints/resnet_best.pth', map_location=device))
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1)
```

## 🔗 Integration with GAN Module

### Feature Extraction for Perceptual Loss

```python
from utils.feature_extractor import load_cnn_feature_extractor

# Load feature extractor (removes final classification layer)
extractor = load_cnn_feature_extractor('cnn/checkpoints/resnet_best.pth')

# Extract features from generated images
features = extractor(generated_images)  # Shape: [batch_size, 512, 1, 1]
```

### Key Integration Points

1. **Checkpoint Path:** `cnn/checkpoints/resnet_best.pth`
2. **Feature Dimension:** 512-dimensional feature vectors
3. **Input Requirements:** 3x32x32 RGB tensors, normalized with CIFAR-10 stats
4. **Model State:** Evaluation mode with frozen weights

## 📊 Model Performance

### Training Statistics

- **Dataset:** CIFAR-10 (50,000 train, 10,000 test images)
- **Architecture:** ResNet-18 (modified for 32x32 inputs)
- **Training Time:** ~10 epochs (adjustable in `train_cnn.py`)
- **Best Accuracy:** Displayed after training completion

### Data Augmentation

- Random crop (32x32 with padding=4)
- Random horizontal flip
- Normalization with CIFAR-10 statistics:
  - Mean: (0.4914, 0.4822, 0.4465)
  - Std: (0.2023, 0.1994, 0.2010)

## 🛠️ Technical Details

### Model Architecture Modifications

1. **Initial Conv Layer:** 3x3 kernel (instead of 7x7) for 32x32 inputs
2. **Max Pooling:** Removed to preserve spatial information
3. **Global Average Pooling:** 4x4 → 1x1 (adapted for CIFAR-10)
4. **Skip Connections:** Standard ResNet-18 residual blocks

### Key Components

- **`BasicBlock`:** Core residual block with 2 convolutions
- **`ResNet18`:** Main model class with 4 stages
- **`ResNet18_cifar10()`:** Factory function for CIFAR-10 configuration

## 🔧 Configuration Options

### Training Parameters (in `train_cnn.py`)

```python
NUM_EPOCHS = 10                    # Training epochs
BATCH_SIZE = 128                   # Batch size for training
INITIAL_LR = 0.1                   # Initial learning rate
MOMENTUM = 0.9                     # SGD momentum
WEIGHT_DECAY = 5e-4               # L2 regularization
```

### Data Paths

```python
CHECKPOINT_DIR = './cnn/checkpoints'     # Model save directory
DATA_ROOT = './cnn/data'                 # Dataset storage
PLOT_PATH = './accuracy_graph_plot_resnet.png'  # Training plot
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:** Reduce batch size in `train_cnn.py`
2. **Checkpoint Not Found:** Run training first to generate weights
3. **Import Errors:** Ensure you're in the CNN directory
4. **Data Download Issues:** Check internet connection and disk space

### Performance Tips

- **GPU Training:** Ensure CUDA is available for faster training
- **Batch Size:** Adjust based on GPU memory (128, 64, 32)
- **Learning Rate:** May need tuning for different batch sizes
- **Data Loading:** Increase `num_workers` for faster data loading

## 📈 Extending the Model

### Adding More Epochs

```python
# In train_cnn.py
NUM_EPOCHS = 50  # Increase from 10
```

### Different Architectures

```python
# Modify models/resnet.py to add:
- ResNet34
- ResNet50
- Custom architectures
```

### Additional Datasets

```python
# Modify utils/datasets.py to support:
- CIFAR-100
- ImageNet subsets
- Custom datasets
```

## 📝 License

This project is for educational and research purposes. Please cite if used in academic work.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Last Updated:** March 2026  
**Version:** 1.0.0
