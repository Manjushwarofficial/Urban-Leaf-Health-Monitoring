# Urban-Leaf-Health-Monitoring
![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Status](https://img.shields.io/badge/status-Prototype-orange)

> Satellite & aerial imagery based system to segment urban vegetation (leaf/green cover), compute vegetation indices, extract features and classify vegetation health (SVM / RandomForest / CNN). Designed for research, prototyping, and small-scale deployment.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Config Example](#config-example)
- [Modeling Notes & Best Practices](#modeling-notes--best-practices)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact & Citation](#contact--citation)

---

## Project Overview
This repository provides tools and example code for:
- ingesting RGB / multispectral satellite or UAV tiles,
- preprocessing (radiometric normalization, resizing, index computation like NDVI),
- semantic segmentation to identify vegetation (U-Net / DeepLab / lightweight variants),
- feature extraction (spectral indices, texture, morphology),
- classification of vegetation health using classical ML (SVM / RandomForest) and deep learning (CNN / transfer learning),
- tile-based inference and time-series change detection.

---

## Features
- Semantic segmentation pipelines (U-Net / DeepLabV3+ compatible)
- Spectral index calculation: NDVI, EVI, SAVI (multispectral support)
- Feature extraction: area, perimeter, GLCM texture, spectral stats
- Classical ML: SVM, RandomForest training wrappers
- Deep learning: Patch-level CNNs with transfer learning (ResNet/EfficientNet)
- Patch sampling, augmentation, tiled inference
- Evaluation metrics and visualization (mIoU, Dice, confusion matrix)
- Reproducible experiment setup and config-driven runs

---
