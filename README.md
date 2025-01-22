# MASK_ML

**Mask Segmentation Machine Learning Project**

![Cutie + MobileSAM Demo](assets/cutie_example.gif)

*Example of the Cutie + MobileSAM pipeline available in demos.*

---

## Overview

MASK_ML is a project focused on experimenting with and implementing various video segmentation models. Building on models like SAM, Cutie, and the newly released SAM2, this project aims to create a comprehensive solution for video segmentation tasks. It includes everything from training pipelines to inference systems, all developed with a strong focus on efficiency and performance.

This repository reflects my journey in implementing core ideas from the field into custom training pipelines for video segmentation.

---

## Downloading Datasets

To download datasets, you can use the `download_datasets.py` script. This script supports downloading datasets from various sources, including COCO and Kaggle.

### COCO Dataset

1. **Run the Script**:
   Execute the `download_datasets.py` script using the following command:
   ```bash
   python mask_ml/utils/download_datasets.py
   ```

2. **Select Dataset**:
   Follow the prompts to select the COCO dataset you want to download. The script will handle the download and extraction process.

### Kaggle Datasets

1. **Set Up Kaggle API**:
   Ensure you have the Kaggle API set up. You can follow the instructions [here](https://www.kaggle.com/docs/api) to set up your Kaggle API credentials.

2. **Run the Script**:
   Execute the `download_datasets.py` script using the following command:
   ```bash
   python mask_ml/utils/download_datasets.py
   ```

3. **Select Dataset**:
   Follow the prompts to select the Kaggle competition or dataset you want to download. The script will handle the download and extraction process.

---

## Using `train.py`

To train a model using the `train.py` script, follow these steps:

1. **Ensure Dependencies are Installed**:
   Make sure you have all the necessary dependencies installed. You can install them using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Training Parameters**:
   Edit the `config/training.yaml` file to set your desired training parameters, such as learning rate, batch size, number of epochs, and other configurations.

3. **Run the Training Script**:
   Execute the `train.py` script using the following command:
   ```bash
   python train.py
   ```

   This will start the training process based on the configurations specified in `config/training.yaml`. The script will log the training progress, save model checkpoints, and generate visualizations as configured.

4. **Monitor Training**:
   During training, you can monitor the progress by checking the output directory specified in the configuration file. This directory will contain logs, model checkpoints, and visualizations.

---


## Training Images examples:

### Autoencoder Tasks

To use the Autoencoder in the training use an AutoEncoder model in the repo: We have `ImageAutoEncoder` defined changed in the training.yaml like this. Remembder to use  an autoEncoder loss function. In this case we are using mse.


```yaml
defaults:
  - _self_
  - datasets: mnist_classification
  - model: image_auto_encoder
  - loss_function: mse
  - override hydra/sweeper: optuna
...

```
example outputs:


 - ![](/assets/reconstruction_batch_0.png)

 - ![](/assets/error_heatmaps_batch_0.png)
 - ![](/assets/step_loss_plot.png)





 ## Referencences:


 Some code references and inspirations of the implementation side   and the conceptual levels:



 transformer.py; The code implementation is using parts of the kaparthy nanogpt: https://github.com/karpathy/nanoGPT/blob/master/model.py


flex_attn_scores.py: This is for using the currently (I hope it doesnt get old) api for PyTorch for calculating efficiently attention variants with nvidia custom kernels. https://github.com/pytorch-labs/attention-gym/tree/main/attn_gym/masks



Unet: Use  a very neat implementation of the Unet architecture .https://github.com/milesial/Pytorch-UNet


