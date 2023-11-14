import os
from utils import *
import random
from commen_utils import *

# 初始化数据
# 把所有数据都读取到，然后

BP_mainFile = r"D:\A_CatRank\test_newWholeINS\Base_Pool"
BP_imgPath = os.path.join(BP_mainFile, "image")
# BP_PseudoPath = os.path.join(BP_mainFile, "pseudo_label_transSalNet")

BP_PseudoPath = os.path.join(BP_mainFile, "pseudo_label_transSalNet")
# BP_PseudoPath = os.path.join(BP_mainFile, "pseudo_label_atsal")
# BP_PseudoPath = os.path.join(BP_mainFile, "pseudo_label_transSalNet")
BP_imgVecPath = os.path.join(BP_mainFile, "imgVec")
Bp_insVecPath = os.path.join(BP_mainFile, "insSplitVec")

bp_imgPathTXT = os.path.join(BP_mainFile, "BP_imgInfo.txt")
bp_allDataNpy = os.path.join(BP_mainFile, "bp_vector.npy")

# test_Data
# test_mainFile = r"D:\A_CatRank\test_newWholeINS\New_data"

# 测试图片
test_mainFile = r"D:\A_CatRank\test_newWholeINS\New_data_test"

test_imgPath = os.path.join(test_mainFile, "image")
test_PseudoPath = os.path.join(test_mainFile, "pseudo_label")
test_imgVecPath = os.path.join(test_mainFile, "imgVec")

# save_PredRank = os.path.join(test_mainFile, "transalnet_2")
# save_PredRank = os.path.join(test_mainFile, "transalnet_")
# save_PredRank = os.path.join(test_mainFile, "transalnet_new_REL_innerRelAll1_intraSim08")
# save_PredRank = os.path.join(test_mainFile, "transalnet_new_REL")
# save_PredRank = os.path.join(test_mainFile, "transalnet_new_REL_intraSim08_first10")
# save_PredRank = os.path.join(test_mainFile, "transalnet_new_REL_intraSim08")
save_PredRank = os.path.join(test_mainFile, "transSalNet_new_REL_intrSim07_intraThe02_FS_5_05")



# save_PredRank = os.path.join(test_mainFile, "eps_new_REL_intrSim08")
# save_PredRank = os.path.join(test_mainFile, "eps_new_REL_intrSim08_intraThe02")
# save_PredRank = os.path.join(test_mainFile, "atsal_new_REL_intrSim08_intraThe02")

# save_PredRank = os.path.join(test_mainFile, "atsal_new_REL_intrSim08_intraThe02")



if not os.path.exists(save_PredRank):  # 判在文件夹如果不存在则创建为文件夹
    os.makedirs(save_PredRank)

Init_Flag = False
if_shuffle = True
# set_firstSearch_per = 5
set_firstSearch_per = 5
# set_secondSearch_per = 1
set_secondSearch_per = 0.5

use_firstFilter = True
use_secondFilter = True
# use_innerImg_relation = True
# use_intraImg_relation = True

data_name = ["", "", ""]

test_Dataset = ["360Cat_NewDataset"]

if Init_Flag == True:
   # 找到所有的有效数据
   basePoolImgPath = InitAllData(BP_imgPath, BP_PseudoPath, BP_imgVecPath, Bp_insVecPath, BP_mainFile)

   # 得到basePool中所有数据
   BP_ImgVecMatrix = get_baseDataNumpy(basePoolImgPath, BP_imgVecPath, BP_mainFile, BP_imgPath)

   # 初始化测试数据的所有数据
   # 找到所有的有效数据
   test_imgPath = Init_TestData(test_mainFile, test_Dataset)

# print("{}".format())

# 直接加载所有BP数据
bp_All_ImgPath, bp_npy = bp_AllImgAndNpy(bp_imgPathTXT, bp_allDataNpy)

if use_firstFilter == False and use_secondFilter == False:
   print("当前使用：{}, blood参考：{}数据， relative参考：{}数据".format("noDistill_testResults",
                                                                     int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01)),
                                                                     int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))))

elif use_firstFilter == True and use_secondFilter == False:
   print("当前使用：{}, blood参考：{}数据， relative参考：{}数据".format("FirstFilter",
                                                                     int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01)),
                                                                     int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))))

elif use_firstFilter == True and use_secondFilter == True:
   print("当前使用：{}, blood参考：{}数据， relative参考：{}数据".format("Whole_config",
                                                                     int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01)),
                                                                     int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))))



for data_index in range(0, len(test_Dataset)):
   # 读取当前数据集中的有效数据
   cur_dataTxTPath = os.path.join(test_mainFile, test_Dataset[data_index] + "_valData" + ".txt")
   cur_Dataset_AllImgs = read_TestDataset(cur_dataTxTPath)
   test_Datasetname = test_Dataset[data_index]
   save_DataSet_results = os.path.join(save_PredRank, test_Dataset[data_index])

   print("当前处理的 {} 数据集共 {} 条有效数据".format(test_Dataset[data_index], len(cur_Dataset_AllImgs)))

   # shuffle数据
   if if_shuffle == True:
      random.shuffle(cur_Dataset_AllImgs)


   # BP_PseudoPath = os.path.join(BP_mainFile, "pseudo_label_eps")
   # BP_imgVecPath = os.path.join(BP_mainFile, "imgVec")
   # Bp_insVecPath = os.path.join(BP_mainFile, "insSplitVec")
   get_testImgCatRankOrder(cur_Dataset_AllImgs, test_mainFile, test_Datasetname, save_DataSet_results,
                           bp_All_ImgPath, bp_npy, set_firstSearch_per, set_secondSearch_per, BP_PseudoPath,BP_imgVecPath,Bp_insVecPath,
                           use_firstFilter, use_secondFilter)