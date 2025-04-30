# Sketch Recognition On Quickdraw

**This project implements and evaluates several deep learning architectures for the task of sketch recognition. It provides a framework to compare models based on different approaches, including sequence-based (CNN-LSTM), modern convolutional networks (ConvNeXt), classic CNNs enhanced with specialized layers (ResNet50), and MLP-based architectures (SketchMLP).**  

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19+-orange)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

# Demo
https://github.com/user-attachments/assets/f381d3f9-ccfc-440c-af3b-654cadc4e25b

![Model Architecture Comparison](docs/arch_comparison.png)  
*Hypothetical architecture diagram - replace with actual implementation visuals*

## 🧠 Models Implemented

This repository includes implementations based on the following papers:

1.  **CNN-LSTM:** Processes sketches as sequences of points, combining CNNs for local feature extraction and LSTMs for temporal dependencies.
    * *Paper:* Kim, Y., & Lee, H. (2019). Stroke-based Sketch Recognition with CNN and LSTM. *International Journal of Contents*, 15(4), 8-14.
    * *DOI:* [10.5392/IJoC.2019.15.4.008](https://doi.org/10.5392/IJoC.2019.15.4.008)
    * *Type:* Sequence-based (Vector)

2.  **ConvNeXt:** A modern, pure convolutional network architecture inspired by Vision Transformers, adapted for sketch recognition (likely treats sketches as raster images).
    * *Paper:* Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. *arXiv preprint arXiv:2201.03545*. (Implementation might be based on adapting the general ConvNeXt architecture for sketch data).
    * *arXiv:* [2201.03545](https://arxiv.org/abs/2201.03545)
    * *Type:* Image-based (Raster)

3.  **SketchMLP-S2:** An MLP-based architecture specifically designed for sketch data, potentially focusing on sequence processing.
    * *Paper:* Lin, J., & Li, J. (2021). SketchMLP: An MLP-based Architecture for Sketch Recognition. *arXiv preprint arXiv:2106.07477*.
    * *arXiv:* [2106.07477](https://arxiv.org/abs/2106.07477)
    * *Type:* Image-based (Raster)
      
4.  **ResNet50:** Uses a ResNet50 backbone.
    * *Type:* Image-based (Raster)

## 🏗️ Architecture Overviews


## **In Summary:**

* For **highest accuracy on static sketch classification**, **ConvNeXt** or **SketchMLP-S2** are likely strong contenders.
* For tasks needing **stroke order/dynamics**, **CNN-LSTM** or potentially **SketchMLP-S2** (if sequence-based) are suitable.
* For **speed/efficiency**, **SketchMLP-S2** might be the best choice, depending on its specific performance.

# Libraries

1. ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
2. ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
3. ![Poetry](https://img.shields.io/badge/Poetry-%233B82F6.svg?style=for-the-badge&logo=poetry&logoColor=0B3D8D)

## ⚙️ Environment Setup

```bash
# Install Poetry (if not exists)
pip install poetry 

# Initialize tensorflow environment
cd Sketch-Rcognition-tf
poetry install --sync
poetry shell

# Initialize torch environment
cd Sketch-Rcognition-torch
poetry install --sync
poetry shell
```
