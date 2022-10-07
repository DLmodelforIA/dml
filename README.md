# Diagnosis of Intracranial Aneurysms in Computed Tomography Angiography Using Deep Learning-Based Detection and Segmentation
## Introduction
  In the present study, we developed a novel 3-dimensional (3D) CNN model for the IA segmentation in CTA images, based on the 3D-Unet architecture. The proposed vessel attention Unet (VA-Unet) architecture is shown in the Supplementary Figure. The model takes CTA images as input and outputs a probability mask that contains the probability of each voxel belonging to an aneurysm or not. There are three main modifications of the VA-Unet compared to the original 3D-Unet. First, we implement a 3D-Unet with 3 levels of down-sampling where the first level is applied to all 3 image dimensions, while the remaining 2 levels are only applied to the in-plane (transverse) dimensions. This choice was necessary as for most CTA images, slice spacing is significantly larger (e.g., 0.625 mm or more) than the transverse in-plane voxel spacing (roughly 0.3 mm). Second, we replaced the convolution block in the original 3D-Unet with a combination of a ResNeXt-SE block and a Conv-SE block, in order to obtain better performance of feature extraction and to avoid vanishing gradients. Additionally, we replaced the deconvolution block with a block with convolution followed by an up-sampling block to avoid checkerboard artifacts. 
  Third, we developed a vessel attention (VA) module and added it to this modified Unet structure. In our experiment, the aneurysm segmentation task was highly correlated with vessel information. Using vessel information, the aneurysm segmentation model could achieve significantly improved performance. Thus, we designed a VA module that learns vessel information and feeds it back to help the model find aneurysms. As shown in the Supplementary Figure, in the VA module, another decoder was followed by the encoder of our U-net structure and was designed to output the logits of vessel segmentation. The logits were down-sampled three times and multiplied to each skip connection node at the decoder of the VA-Unet. 
  In the training stage, the CTA images were cropped into patches with a size of [32,224,224]. Hounsfield units (HU) in each patch were clipped to the range [-1024, 2048] and then normalized to [0,1]. Data augmentations including shifting, flipping, rotation, and zooming were used to increase the training data. The explog loss function was used for both the segmentation and VA modules. The sum of the two losses was used as the total loss, where the loss of the VA module was multiplied by a factor of 0.1. The network was trained by minimizing total loss using the Adam Optimizer. The size of mini-batch was set to 4. In each mini-batch, both positive and negative patches were sampled. Learning rate was set to 1e-4 with warm-up and cosine decay. Model embedding was used in the proposed VA-Unet (supplementary figure 1). Aiming to speed up our model, we used a larger input size of [48,256,256] in the inference stage than the size of [32,224,224] in the training stage, since 3D-Unet is not limited by input size. 


Prerequisites
* Ubuntu 18.04 lts
* NVIDIA GPU + CUDA_10.0 CuDNN_7.5
* Python 3.8.5
* tensorflow-gpu 1.14.0

This inplementation has been tested on NVIDIA V100 32GB and NVDIA Tesla P100 16GB. 

## Installation

Other packages are as follows:
* json
* yaml
* scipy
* SimpleITK
* scikit-image
* numpy

Install dependencies:
```shell script
pip install -r requirements.txt
```

## Dataset
```shell script
Patient_ID_Folder
├── image.nii.gz  # original image
├── aneurysm_mask.nii.gz   # the mask of aneurysm
├── vessel_mask.nii.gz  # the mask of vessel
```

## Usage

### Train
Run command as below.
```shell script
python train.py --config configs/aneurysm_seg.yaml
```
* `train.epochs`: number of epochs 
* `train.batch_size`: batch size of training stage
* `train.train_file`: the list of train data's filenames
* `train.valid_file`: the list of validation data's filenames


### Inference
the checkpoint file are stored at: ```ckpt_output```

configuration file: `configs/aneurysm_seg.yaml`
* `inference.test_file`: the json file path where the data to be tested is stored.
* `inference.batch_size`: the batch size of test stage

Run command as below.
```shell script
python inference.py --config configs/aneurysm_seg.yaml --ckpt ckpt_output/segan_net.ckpt-24
```


