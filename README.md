## This repository contains the code for submission paper
# CR360
 
------
 
&ensp Noticed that our proposed `unsupervised approach` based on knowledge transfer and data mining mechieasm. Therefore, it is necessary to prepare some unlabeled panoramic data and some data you would like predicted their catergory-level ranking info.

## Table of Contents:
> * Requirements
> * Main Steps
> * Detailed procedure of Code
> * CR-360 dataset

## 1.Requirements:
> * Python 3.7.4
> * CUDA 11.6.2
> * Opencv python

## 2.Main Steps:
1. `Data_preprocess` --> 2. `CR_knowledgeDist` --> 3. `Fusion-Net` --> 4. `Evaluate`
 
## 3. Detailed Procedure:
* #### `Data_preprocess`
&nbsp;&nbsp; DETR or any other object detector tools are required to generate semantic features for raw unlabeled data. And Please save the bounding boxes info and semantic embedding for each instances in image. Then run the `EncodeForEmbed.py` to yield image-level features and `sal_toInsRank.py` to get Pesudo Instance-level labels.
&nbsp;&nbsp; If you don't have a large amount unlabeled data, we also prepare our raw dataset here: [Google Drive](), including matched pseudo labels and the extracted features of images and instances.

* #### `CR_knowledgeDist`
Prepare new scenes and follow former step to extract semantic features. Then run the `main.py` to generate knowledge matrices and save the distilled matrices to '.npy' format.

* #### `Fusion-Net`
For train:


For test:
Choose any Fusion-net in the folder, and run `my_predict.py` to generate final category-level rank.

* #### `Evaluate`
If you want to evaluate you results, run `new_sor.py` and change the input_path to you own.

## 4. CR-360 dataset:
&nbsp;&nbsp; Since there is no related dataset we could adopt ot access our proposed paradigm, a brand-new `CR-360` dataset is constructed with total 557 panoramic images with dence instance-level masks among 151 catergories. Besides, we also provied robust category-lavel ranking labels manually annootated by 13 volunteers.  
&nbsp;&nbsp; And we have released this datdase in : [Google Drive](/figure/class.svg)
<div align="center"><img src="/figure/class.svg" width="500" height="" alt="抖音小程序"/><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Categories of our proppsed CR-360 dataset.


