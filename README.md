## This repository contains the code for submission paper
# CR360
 
------
 
Noticed that our proposed `unsupervised approach` based on knowledge transfer and data mining mechieasm. Therefore, it is necessary to prepare some unlabeled panoramic data and some data you would like predicted their catergory-level ranking info.

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
DETR or any other object detector tools are required to generate semantic features for raw unlabeled data. And Please save the bounding boxes info and semantic embedding for each instances in image. Then run the `EncodeForEmbed.py` to yield image-level features and `sal_toInsRank.py` to get Pesudo Instance-level labels.

* #### `CR_knowledgeDist`
Prepare new scenes and follow former step to extract semantic features. Then run the `main.py` to generate knowledge matrices and save the distilled matrices to '.npy' format.

* #### `Fusion-Net`
Choose any Fusion-net in the fold, and ran `my_predict.py` to generate final category-level rank.

* #### `Evaluate`
If you want to evaluate you results, run `new_sor.py` and change the input_path to you own.

## 4. CR-360 dataset:
Since there is no related dataset we could adopt ot access our proposed paradigm, a brand-new `CR-360` dataset is constructed with total 557 panoramic images with dence instance-level masks among 151 catergories. Besides, we also provied robust category-lavel ranking labels manually annootated by 13 volunteers.
And we have released this datdase in : [Google Drive](https://drive.google.com/file/d/1UZ1PQvbHXVUF2HskD1xYAXVIM5JsenYm/view?usp=drive_link)




