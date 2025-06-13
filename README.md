# 🧠 ResNet-18 on CIFAR-10

This repository implements a **ResNet-18** Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset using PyTorch.

ResNet-18 is a residual neural network architecture designed to address the **vanishing gradient problem** in deep networks by introducing **skip connections**. It has 18 layers and is widely used for image classification tasks due to its balance between depth and efficiency.

---

## 📊 Dataset: CIFAR-10

- 60,000 images (32x32 color)
- 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- 50,000 training images + 10,000 test images

---

## 🏗️ Model Architecture

ResNet-18 includes:
- Initial convolution + batch normalization + ReLU
- 4 residual blocks (with downsampling)
- Global average pooling
- Fully connected classification layer

---

## 🚀 Training Details

| Epoch | Test Accuracy (%) | Notable Observations |
|-------|-------------------|----------------------|
| 0     | 46.18             | High loss (~2.45 to 1.3); model starting to learn |
| 1     | 59.95             | Significant accuracy jump as model converges |
| 2     | 67.09             | Stable improvements, lower loss |
| 3     | 73.11             | Less fluctuation in loss; signs of generalization |
| 4     | 74.82             | Further improved accuracy, smoother loss |
| 5     | 77.71             | Steady training with slight variance in batch loss |
| 6     | 78.61             | Small improvements, some overfitting signs begin |
| 7     | 80.85             | Best accuracy so far, resilient across batches |
| 8     | 79.34             | Slight drop; possibly due to noise or learning rate |
| 9     | **81.82**         | ✅ Final peak accuracy |

---

## 🧮 Model Parameters

- **Total Trainable Parameters**: `11,689,512`


