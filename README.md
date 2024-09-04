
# MASK_ML

**Mask Segmentation Machine Learning Project**

![Cutie + MobileSAM Demo](assets/cutie_example.gif)

*Example of the Cutie + MobileSAM pipeline available in demos.*

---

## Overview

MASK_ML is a project focused on experimenting with and implementing various video segmentation models. Building on models like SAM, Cutie, and the newly released SAM2, this project aims to create a comprehensive solution for video segmentation tasks. It includes everything from training pipelines to inference systems, all developed with a strong focus on efficiency and performance.

This repository reflects my journey in implementing core ideas from the field into custom training pipelines for video segmentation.

---

## Project Principles

To ensure the project's efficiency and scalability, the following principles guide its development:

1. **Minimize Dependencies**:  
   Every new dependency must be carefully considered. Only absolutely necessary dependencies should be added to keep the project self-contained and lightweight. The primary dependencies are:
   - **PyTorch**: All implementations will be built using PyTorch. If a function is available in PyTorch, it will be prioritized over alternatives like OpenCV or NumPy to maintain consistency. Ideally, all data structures will remain as tensors.
   - **OpenCV**: Used exclusively for video visualizations, transformations, and device connections. Numpy will only be used when necessary for image annotations or device I/O.

2. **Performance Optimization**:  
   PyTorch is already fast, but further optimization will be explored, such as model quantization and conversions to ONNX. While the PyTorch inference pipeline should be optimized natively, ONNX Runtime support will be added for flexibility in deployment.

3. **Interactive & Modular**:  
   The project will adopt a library-like structure, with individual scripts demonstrating various parts of the pipeline. Demos will showcase different functionalities for end-to-end testing.

4. **Easy Configuration for Training & Testing**:  
   To streamline the process, the project will support easy configuration options. This includes:
   - **Python `setup.py` configurations** for traditional setups.
   - **Hydra-core** integration for modular and dynamic configuration management.
   - **Docker pipelines** for containerized and reproducible environments.

---



## Future Work

More features are under active development! Follow the commit history to stay updated on the latest progress.

---

## Getting Started

Instructions for setting up the project, installing dependencies, and running the training and inference scripts will be provided here soon.

---

Thank you for checking out the MASK_ML project! Contributions and feedback are always welcome.