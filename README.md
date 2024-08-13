This is the start of a project that will try to experiment with Video Object Segmentation. The idea is to build the Software tools for experimenting with Video Object Segmentation. Ideally we will want to replicate Cutie and SAM2. Create code to have experiments with it. The core idea is not to use only jupyter_notebooks have benchmarks  and Applications for testing different ideas and Ideally create something new! The race begin in here and hopefully it will be a fun experience.


First Ideas to implement:

1. The first thing to implement is a simple VIT head  with a segmentation objective. 

2. Create a training script only using Pytorch for training that VIT for a simple segmentation task

3. Create scripts for downloading different segmentation task datasets.

4. Create scripts to visualize results. Ideally The original image with the segmentation.

5. Create Validation testing and test scores.

CORE VALUES of the Project:

This are the things that we have to respect when building this project. First of all it should have as less depdencies as possible. Eveytime time we want to add a new dependency we must really evaluate why we are adding it and if it is completely necessary. This goes with the idea that we must create the application as most self contained as possible.  Dependencies space in memory should be as more reduced as possible. Core dependencies we will need. Pytorch, ideally all the code and implementation will be coded in Pytorch. This means if there is a function that is implemented in OpenCv , numpy, Simpy and Pytorch we will always select the Pytorch version. Ideally all arrays should be Tensors. For Video Visualizations and Transformations OpenCV as a lot functions that will make the work way more easirer. Specially for connecting with Devices and Image annotations. Ideally that should be the only reason we convert things to numpy.


Pytorch is already fast!! Later on the project we will try to optimize the models, using Quantization, ONNX RUNTIME. We dont want to make it fast using INNXRUNTIME the Pytorch Inference version should already be fast. But we also want to be able to convert to ONNX.


It should be interactive and have library structure. In the same way that SDKs have demos for their different functionalities. We should have scripts that gives information of the different parts of the Pipeline. End to End testing in a apllication


Easy Configurations for Training and testing. This means either going for ass pain python setup.py configurations , using hydra-core or dockers pipelines. 
# MASK_ML
