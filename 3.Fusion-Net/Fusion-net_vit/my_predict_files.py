import os
import json
from utils_metric import ALL_metricCompute, ALL_metricCompute222
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model.fuisonNet import FusionNet
from utils import get_predInfo, COCO_cat, outPut_rankLabel, ouputINFO, writeInInfo
import numpy as np




def make_input(test_npyPath, img_name):
    img_name = img_name.split(".")[0]
    train_blood_inner = os.path.join(test_npyPath, "blood", "inner_npy")
    train_blood_intra = os.path.join(test_npyPath, "blood", "intra_npy")
    train_relative_inner = os.path.join(test_npyPath, "relative", "inner_npy")
    train_relative_intra = os.path.join(test_npyPath, "relative", "intra_npy")

    cur_blood_inner = os.path.join(train_blood_inner, img_name + ".npy")
    cur_blood_intra = os.path.join(train_blood_intra, img_name + ".npy")
    cur_relative_inner = os.path.join(train_relative_inner, img_name + ".npy")
    cur_relative_intra = os.path.join(train_relative_intra, img_name + ".npy")

    blood_inner = np.load(cur_blood_inner)
    blood_intra = np.load(cur_blood_intra)
    relative_inner = np.load(cur_relative_inner)
    relative_intra = np.load(cur_relative_intra)

    blood_inner = blood_inner.reshape((1, 80, 80))
    blood_intra = blood_intra.reshape((1, 80, 80))
    relative_inner = relative_inner.reshape((1, 80, 80))
    relative_intra = relative_intra.reshape((1, 80, 80))

    blood_matrix = np.concatenate((blood_inner, blood_intra), axis=0)
    relative_matrix = np.concatenate((relative_inner, relative_intra), axis=0)

    blood_matrix = torch.tensor(blood_matrix)
    relative_matrix = torch.tensor(relative_matrix)
    # float 类型
    blood_matrix = blood_matrix.float()
    relative_matrix = relative_matrix.float()

    return blood_matrix, relative_matrix

# 使用DETR自带的类别
use_DETR = True
# DETR_catPath = r"E:\final_project\rank_project\test_newWhole\New_data\pseudo_label_PredictLabel\360Cat_NewDataset\1"
DETR_catPath = r"E:\final_project\360-test-557\other_files\dection\DETR_CatCount"

def main(cur_sorBest, test_Path, save_infoPath):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         ])

    # create model
    backbone_type = "unet"
    model = FusionNet(input_channel=5, backbone_type=backbone_type)
    # load model weights
    # weights_path = "./vgg16Net.pth"
    weights_path = cur_sorBest
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    test_npyPath = test_Path
    img_names = os.listdir(os.path.join(test_npyPath, "blood", "inner_npy"))

    GT_txtPath = r"E:\final_project\360-test-557\NEW_ANNA\whole_allMask\Z_FinalDataset\GT_COCO_catRank"
    # save_predResult = r"D:\A_CatRank\test_newWholeINS\New_data_test\atsal_new_REL_intraSim08\save_path"

    GT_pathList = []
    Pred_NameList = []
    Pred_ValueList = []

    model.eval()
    with torch.no_grad():
        for img_index in range(0, len(img_names)):
            # npy_path = os.path.join(test_npyPath, img_names[img_index])
            cur_imgGTPath = os.path.join(GT_txtPath, img_names[img_index].split(".")[0] + ".txt")
            img_name = img_names[img_index].split(".")[0]

            blood_matrix, relative_matrix = make_input(test_npyPath, img_names[img_index])

            # print(blood_matrix.shape)
            # print(relative_matrix.shape)
            blood_matrix = blood_matrix.unsqueeze(dim=0)
            relative_matrix = relative_matrix.unsqueeze(dim=0)
            # print(blood_matrix.shape)
            # print(relative_matrix.shape)
            # print("{}".format())
           #  print()
            output = torch.squeeze(model(blood_matrix.to(device), relative_matrix.to(device))).cpu()
            Pred_NameOrder, Pred_ValueOrder = outPut_rankLabel(output, cur_imgGTPath)

            GT_pathList.append(cur_imgGTPath)
            Pred_NameList.append(Pred_NameOrder)
            Pred_ValueList.append(Pred_ValueOrder)

            # writeInInfo(save_predResult, img_name, Pred_NameOrder, Pred_ValueOrder)

    sor, tt, top_1, str_list = ALL_metricCompute(GT_pathList, Pred_NameList, Pred_ValueList)
    print("使用GT类别")
    ouputINFO(str_list)
    # sor, tt, top_1, str_list = ALL_metricCompute222(GT_pathList, Pred_NameList, Pred_ValueList, use_DETR, DETR_catPath)
    # print(str_list)
    print("使用DETR类别")
    ouputINFO(str_list)


    top_1_str = str_list[-1]
    # print(top_1_str)
    # print("{}".format())
    # save_infoPath
    f = open(save_infoPath, 'a')
    f.write(top_1_str)
    f.close()
    # print("{}".format())


if __name__ == '__main__':

    weight_file = r"E:\final_project\weights\Vit"
    test_file = r"D:\A_CatRank\test_newWholeINS\test_file"

    source_names = os.listdir(test_file)

    for name_index in range(0, len(source_names)):
        cur_name = source_names[name_index]
        source_path = os.path.join(weight_file, cur_name)
        source_setNames = os.listdir(source_path)

        test_wayPath = os.path.join(test_file, cur_name)
        for set_index in range(0, len(source_setNames)):
            cur_Setpath = os.path.join(source_path, source_setNames[set_index])
            aug_files = os.listdir(cur_Setpath)
            # \360Cat_NewDataset\whole_testResults
            test_setPath = os.path.join(test_wayPath, source_setNames[set_index], "360Cat_NewDataset", "whole_testResults")

            for aug_index in range(0, len(aug_files)):
                cur_augPath = os.path.join(cur_Setpath, aug_files[aug_index])
                cur_sorBest = os.path.join(cur_augPath, "best_sor.pth")

                print(cur_sorBest)
                print(test_setPath)
                save_infoPath = os.path.join(cur_augPath, "metric.txt")
                main(cur_sorBest, test_setPath, save_infoPath)