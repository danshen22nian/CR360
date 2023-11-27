import os
from utils import *
import random
from commen_utils import *

BP_mainFile = r"E:\our_rawData\raw_dataset"
BP_imgPath = os.path.join(BP_mainFile, "image")
BP_PseudoPath = os.path.join(BP_mainFile, "pseudo_label")
BP_imgVecPath = os.path.join(BP_mainFile, "imgVec")
Bp_insVecPath = os.path.join(BP_mainFile, "insSplitVec")
bp_imgPathTXT = os.path.join(BP_mainFile, "BP_imgInfo.txt")
bp_allDataNpy = os.path.join(BP_mainFile, "bp_vector.npy")

test_mainFile = r"E:\our_rawData\test_dataset"

test_imgPath = os.path.join(test_mainFile, "image")
test_PseudoPath = os.path.join(test_mainFile, "pseudo_label")
test_imgVecPath = os.path.join(test_mainFile, "imgVec")

save_PredRank = os.path.join(test_mainFile, "matrix_save")
if not os.path.exists(save_PredRank):
    os.makedirs(save_PredRank)

Init_Flag = True
if_shuffle = True

set_firstSearch_per = 5

set_secondSearch_per = 0.5

use_firstFilter = True
use_secondFilter = True

test_Dataset = ["CR_360"]

if Init_Flag == True:

   basePoolImgPath = InitAllData(BP_imgPath, BP_PseudoPath, BP_imgVecPath, Bp_insVecPath, BP_mainFile)


   BP_ImgVecMatrix = get_baseDataNumpy(basePoolImgPath, BP_imgVecPath, BP_mainFile, BP_imgPath)

   test_imgPath = Init_TestData(test_mainFile, test_Dataset)


bp_All_ImgPath, bp_npy = bp_AllImgAndNpy(bp_imgPathTXT, bp_allDataNpy)

if use_firstFilter == False and use_secondFilter == False:
   print("current：{}, blood_num：{} ， relative_num：{}".format("noDistill_testResults",
                                                                     int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01)),
                                                                     int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))))

elif use_firstFilter == True and use_secondFilter == False:
   print("current：{}, blood_num：{}， relative_num：{}".format("FirstFilter",
                                                                     int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01)),
                                                                     int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))))

elif use_firstFilter == True and use_secondFilter == True:
   print("current：{}, blood_num：{}， relative_num：{}".format("Whole_config",
                                                                     int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01)),
                                                                     int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))))



for data_index in range(0, len(test_Dataset)):

   cur_dataTxTPath = os.path.join(test_mainFile, test_Dataset[data_index] + "_valData" + ".txt")
   cur_Dataset_AllImgs = read_TestDataset(cur_dataTxTPath)
   test_Datasetname = test_Dataset[data_index]
   save_DataSet_results = os.path.join(save_PredRank, test_Dataset[data_index])

   print("current_dataset {} total_datasize: {}".format(test_Dataset[data_index], len(cur_Dataset_AllImgs)))

   if if_shuffle == True:
      random.shuffle(cur_Dataset_AllImgs)
   get_testImgCatRankOrder(cur_Dataset_AllImgs, test_mainFile, test_Datasetname, save_DataSet_results,
                           bp_All_ImgPath, bp_npy, set_firstSearch_per, set_secondSearch_per, BP_PseudoPath,BP_imgVecPath,Bp_insVecPath,
                           use_firstFilter, use_secondFilter)