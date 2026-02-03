# Solar Panel Surface Fault Classification (Deep Learning)

## Overview
This project applies deep learning and computer vision techniques to automatically classify surface conditions of solar panels. The objective is to support scalable quality control and maintenance by detecting common faults that reduce solar panel efficiency.

A transfer learning approach using MobileNetV2 is employed, and both baseline and fine-tuned models are evaluated to study generalization performance.


## Problem Statement
Surface faults such as dust accumulation, snow coverage, bird droppings, and structural damage can significantly reduce solar panel performance. Manual inspection is costly and difficult to scale for large solar farms.

This project investigates whether deep learning models can reliably classify these surface faults from images.


## Dataset
- 885 images across 6 classes:
  - Clean  
  - Dusty  
  - Snow-Covered  
  - Bird-drop  
  - Physical-Damage  
  - Electrical-Damage  
- 80/20 training–validation split  
- Dataset is **imbalanced, reflecting real-world fault distributions  



## Approach
- Transfer learning using MobileNetV2 pretrained on ImageNet  
- Baseline model with frozen backbone  
- Fine-tuning experiment for comparison  
- Model selection based on validation performance  



## Key Results
- Baseline model: ~68% validation accuracy  
- Fine-tuned model: Lower validation accuracy due to overfitting  
- Baseline model selected for **better generalization  

Detailed training curves, confusion matrices, and evaluation metrics are available in the notebook.

**See full implementation:** `solar_fault_classifier.ipynb`



## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  



