## This repository contains the code for submission paper
# CR360
 
------
 
Noticed that our proposed `unsupervised approach` based on knowledge transfer and data mining mechanism. Therefore, it is necessary to prepare some unlabeled panoramic data and test data you would like predicted their catergory-level ranking info.

## Table of Contents:
> * **Requirements**
> * **Main Steps**
> * **Detailed procedure of Code**
> * **CR-360 dataset**

## 1.Requirements:
> * **Python 3.7.4**
> * **CUDA 11.6.2**
> * **Opencv python**
> * **pytorch**

## 2.Main Steps:
**1. `Data_preprocess`** --> **2. `CR_knowledgeDist`** --> **3. `Fusion-Net`**
 
## 3. Detailed Procedure:
* #### `Data_preprocess`
&nbsp;&nbsp; DETR or any other object detector tools are required to generate semantic features for raw unlabeled data. And Please save the bounding boxes info and semantic embedding for each instances in image. Then run the `EncodeForEmbed.py` to yield image-level features and `sal_toInsRank.py` to get Pesudo Instance-level labels.  
&nbsp;&nbsp; If you don't have a large amount unlabeled data, we also prepare our raw dataset here: [Google Drive](https://drive.google.com/file/d/11xMs3l3ylyZk9JPYp9Ko4QABkn_UCrCB/view?usp=sharing), including matched pseudo IR labels and the extracted features of images and instances.

* #### `CR_knowledgeDist`
Prepare new scenes and follow former step to extract semantic features. Then run the `main.py` to generate knowledge matrices and save the distilled matrices to '.npy' format. The prodeuced matrices will be used to train Fusion-Net later.

* #### `Fusion-Net`
**For train:**
Change the input path and label path to your own. Then, choose any Fusion-net in the folder, and run `train.py` to generate final category-level rank.

**For test:**
Choose any Fusion-net in the folder, and run `my_predict.py` to generate final category-level rank. Or we preapre a model weight with TranSalNet on our Unet Fusion-Net here:  
[Google Drive](https://drive.google.com/file/d/1QaYl_L8E1kmSflcByOGa-jxjx8dPhf-8/view?usp=sharing).  
[Baidu Drive](https://pan.baidu.com/s/1luAosjqW2piFyxvf42KXJA) 提取码：8lyn


