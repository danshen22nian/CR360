import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class_str = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light," \
            "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear," \
            "zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite," \
            "baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon," \
            "bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table," \
            "toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors," \
            "teddy bear,hair drier,toothbrush"

COCO_cat = []
all_class = class_str.split(",")
for i in range(0, len(all_class)):
    cur_cat = all_class[i]
    COCO_cat.append(cur_cat)


def read_split_data2(train, val, GT):
    # train blood_inner, blood_intra
    train_blood_inner = os.path.join(train, "blood", "inner_npy")
    train_blood_intra = os.path.join(train, "blood", "intra_npy")
    train_relative_inner = os.path.join(train, "relative", "inner_npy")
    train_relative_intra = os.path.join(train, "relative", "intra_npy")

    train_npy = []
    train_txt = []
    if len(os.listdir(train_blood_inner)) == len(os.listdir(train_blood_intra)) == len(os.listdir(train_relative_inner)) == len(os.listdir(train_relative_intra)):
        train_names = os.listdir(train_blood_inner)
        for img_index in range(0, len(train_names)):

            cur_blood_inner = os.path.join(train_blood_inner, train_names[img_index])
            cur_blood_intra = os.path.join(train_blood_intra, train_names[img_index])
            cur_relative_inner = os.path.join(train_relative_inner, train_names[img_index])
            cur_relative_intra = os.path.join(train_relative_intra, train_names[img_index])

            train_npy.append([(cur_blood_inner, cur_blood_intra), (cur_relative_inner, cur_relative_intra)])

            lebel_Path = os.path.join(GT, train_names[img_index].split(".")[0] + ".txt")
            train_txt.append(lebel_Path)

    val_blood_inner = os.path.join(val, "blood", "inner_npy")
    val_blood_intra = os.path.join(val, "blood", "intra_npy")
    val_relative_inner = os.path.join(val, "relative", "inner_npy")
    val_relative_intra = os.path.join(val, "relative", "intra_npy")
    val_npy = []
    val_txt = []
    if len(os.listdir(val_blood_inner)) == len(os.listdir(val_blood_intra)) == len(os.listdir(val_relative_inner)) == len(os.listdir(val_relative_intra)):
        val_names = os.listdir(val_blood_inner)
        for img_index in range(0, len(val_names)):
            cur_blood_inner = os.path.join(val_blood_inner, val_names[img_index])
            cur_blood_intra = os.path.join(val_blood_intra, val_names[img_index])
            cur_relative_inner = os.path.join(val_relative_inner, val_names[img_index])
            cur_relative_intra = os.path.join(val_relative_intra, val_names[img_index])

            val_npy.append([(cur_blood_inner, cur_blood_intra), (cur_relative_inner, cur_relative_intra)])

            lebel_Path = os.path.join(GT, val_names[img_index].split(".")[0] + ".txt")
            val_txt.append(lebel_Path)


    return train_npy, train_txt, val_npy, val_txt



def read_split_data3(main_train, val, GT):

    all_ways = os.listdir(main_train)
    train_npy = []
    train_txt = []
    for i in range(0, len(all_ways)):
        train = os.path.join(main_train, all_ways[i], "360Cat_NewDataset", "whole_testResults")

        train_blood_inner = os.path.join(train, "blood", "inner_npy")
        train_blood_intra = os.path.join(train, "blood", "intra_npy")
        train_relative_inner = os.path.join(train, "relative", "inner_npy")
        train_relative_intra = os.path.join(train, "relative", "intra_npy")

        if len(os.listdir(train_blood_inner)) == len(os.listdir(train_blood_intra)) == len(os.listdir(train_relative_inner)) == len(os.listdir(train_relative_intra)):
            train_names = os.listdir(train_blood_inner)
            for img_index in range(0, len(train_names)):
                # input_Path = os.path.join(train, train_names[img_index])
                cur_blood_inner = os.path.join(train_blood_inner, train_names[img_index])
                cur_blood_intra = os.path.join(train_blood_intra, train_names[img_index])
                cur_relative_inner = os.path.join(train_relative_inner, train_names[img_index])
                cur_relative_intra = os.path.join(train_relative_intra, train_names[img_index])

                train_npy.append([(cur_blood_inner, cur_blood_intra), (cur_relative_inner, cur_relative_intra)])

                lebel_Path = os.path.join(GT, train_names[img_index].split(".")[0] + ".txt")
                train_txt.append(lebel_Path)

    val_blood_inner = os.path.join(val, "blood", "inner_npy")
    val_blood_intra = os.path.join(val, "blood", "intra_npy")
    val_relative_inner = os.path.join(val, "relative", "inner_npy")
    val_relative_intra = os.path.join(val, "relative", "intra_npy")
    val_npy = []
    val_txt = []
    if len(os.listdir(val_blood_inner)) == len(os.listdir(val_blood_intra)) == len(os.listdir(val_relative_inner)) == len(os.listdir(val_relative_intra)):
        val_names = os.listdir(val_blood_inner)
        for img_index in range(0, len(val_names)):
            cur_blood_inner = os.path.join(val_blood_inner, val_names[img_index])
            cur_blood_intra = os.path.join(val_blood_intra, val_names[img_index])
            cur_relative_inner = os.path.join(val_relative_inner, val_names[img_index])
            cur_relative_intra = os.path.join(val_relative_intra, val_names[img_index])

            val_npy.append([(cur_blood_inner, cur_blood_intra), (cur_relative_inner, cur_relative_intra)])

            lebel_Path = os.path.join(GT, val_names[img_index].split(".")[0] + ".txt")
            val_txt.append(lebel_Path)


    return train_npy, train_txt, val_npy, val_txt



class_str = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light," \
            "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear," \
            "zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite," \
            "baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon," \
            "bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table," \
            "toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors," \
            "teddy bear,hair drier,toothbrush"
COCO_cat = []
all_class = class_str.split(",")
for i in range(0, len(all_class)):
    cur_cat = all_class[i]
    COCO_cat.append(cur_cat)

def get_predInfo(path):

    labelInfo = open(path, "r", encoding="UTF-8")
    # 所有测试数据的信息
    labelInfo_Lines = labelInfo.readlines()
    labelInfo.close()

    order = labelInfo_Lines[-1].split("!")[-1]
    order = order.split("{")[1].split("}")[0].split(",")

    object_name = []
    prior_score = []

    if len(labelInfo_Lines)==1:
        return object_name, prior_score
    for i in range(0, len(order)):
        img_class = order[i].strip()
        if i != len(order) - 1:
            class_name = img_class.split("'")[1]
            score = float(img_class.split(":")[1].strip())
        else:
            class_name = img_class.split("'")[1]
            score = float(img_class.split(":")[1].strip())

        object_name.append(class_name)
        prior_score.append(score)

    return object_name, prior_score


def outPut_rankLabel(output, img_GTPath):

    output = output.tolist()
    GT_allCats, prior_score = get_predInfo(img_GTPath)

    cat_toValue = []

    for i in range(0, len(GT_allCats)):
        cur_cat = GT_allCats[i]
        cat_index = COCO_cat.index(cur_cat)
        cat_toValue.append(output[cat_index])

    Pred_dict = dict(zip(GT_allCats, cat_toValue))
    Pred_dict = sorted(Pred_dict.items(), key=lambda item: item[1], reverse=True)

    Pred_NameOrder, Pred_ValueOrder = [], []

    for i in range(0, len(Pred_dict)):
        cur_name, cur_value = Pred_dict[i]
        Pred_NameOrder.append(cur_name)
        Pred_ValueOrder.append(cur_value)

    return Pred_NameOrder, Pred_ValueOrder


def ouputINFO(list):
    print("================================================")
    for i in range(0, len(list)):
        print(list[i])
    print("================================================")



def writeInInfo(save_predResult, img_name, Pred_NameList, Pred_ValueList):
    save_path = os.path.join(save_predResult, img_name + ".txt")
    for i in range(0, len(Pred_NameList)):
        line_str = f"{Pred_NameList[i]}, PriorScore:{Pred_ValueList[i]}\n"
        f = open(save_path, 'a')
        f.write(line_str)
        f.close()

    Pred_dict = dict(zip(Pred_NameList, Pred_ValueList))

    str = f"FinalOrder!{Pred_dict}\n"
    f = open(save_path, 'a')
    f.write(str)
    f.close()

def writeINFO(Info, best_Epoch, save_metricINFO):
    title_str = "Best SOR IN epoch {}".format(best_Epoch) + "\n"
    f = open(save_metricINFO, 'a')
    f.write(title_str)
    f.close()
    for i in range(0, len(Info)):
        line_str = f"{Info[i]}\n"
        f = open(save_metricINFO, 'a')
        f.write(line_str)
        f.close()


def matrix_shuffle(ori_matrix, ori_list):
    shuffle_list = ori_list.copy()
    random.shuffle(shuffle_list)

    height, width = ori_matrix.shape
    shuffle_matrix = np.zeros_like(ori_matrix)
    for i in range(0, height):
        i_indexIN_oriMatrix = ori_list.index(shuffle_list[i])
        for j in range(0, width):
            j_indexIN_oriMatrix = ori_list.index(shuffle_list[j])
            shuffle_matrix[i][j] = ori_matrix[i_indexIN_oriMatrix, j_indexIN_oriMatrix]
    return shuffle_matrix, shuffle_list


def concat_npy(cur_npyPath):

    blood_inner = np.load(cur_npyPath[0][0])
    blood_intra = np.load(cur_npyPath[0][1])
    relative_inner = np.load(cur_npyPath[1][0])
    relative_intra = np.load(cur_npyPath[1][1])

    blood_inner = blood_inner.reshape((1, 80, 80))
    blood_intra = blood_intra.reshape((1, 80, 80))
    relative_inner = relative_inner.reshape((1, 80, 80))
    relative_intra = relative_intra.reshape((1, 80, 80))

    blood_matrix = np.concatenate((blood_inner, blood_intra), axis=0)
    relative_matrix = np.concatenate((relative_inner, relative_intra), axis=0)

    return blood_matrix, relative_matrix

def matrix_augment(cur_npyPath, ori_label, ori_Catlist, times):

    blood_inner = np.load(cur_npyPath[0][0])
    blood_intra = np.load(cur_npyPath[0][1])
    relative_inner = np.load(cur_npyPath[1][0])
    relative_intra = np.load(cur_npyPath[1][1])


    shuffle_bloodList, shuffle_relativeList, shuffle_labelList = [], [], []

    for index in range(0, times):
        shuffle_list = ori_Catlist.copy()
        random.shuffle(shuffle_list)

        height, width = blood_inner.shape
        shuffle_blood_inner = np.zeros_like(blood_inner)
        shuffle_blood_intra = np.zeros_like(blood_intra)
        shuffle_relative_inner = np.zeros_like(relative_inner)
        shuffle_relative_intra = np.zeros_like(relative_intra)
        for i in range(0, height):
            i_indexIN_oriMatrix = ori_Catlist.index(shuffle_list[i])
            for j in range(0, width):
                j_indexIN_oriMatrix = ori_Catlist.index(shuffle_list[j])
                shuffle_blood_inner[i][j] = blood_inner[i_indexIN_oriMatrix, j_indexIN_oriMatrix]
                shuffle_blood_intra[i][j] = blood_intra[i_indexIN_oriMatrix, j_indexIN_oriMatrix]
                shuffle_relative_inner[i][j] = relative_inner[i_indexIN_oriMatrix, j_indexIN_oriMatrix]
                shuffle_relative_intra[i][j] = relative_intra[i_indexIN_oriMatrix, j_indexIN_oriMatrix]
        shuffle_label = np.zeros_like(ori_label)

        for i in range(0, len(ori_Catlist)):

            cur_catIndex = shuffle_list.index(ori_Catlist[i])
            shuffle_label[cur_catIndex] = ori_label[i]

        shuffle_blood_inner = shuffle_blood_inner.reshape((1, 80, 80))
        shuffle_blood_intra = shuffle_blood_intra.reshape((1, 80, 80))
        shuffle_relative_inner = shuffle_relative_inner.reshape((1, 80, 80))
        shuffle_relative_intra = shuffle_relative_intra.reshape((1, 80, 80))

        shuffle_blood_matrix = np.concatenate((shuffle_blood_inner, shuffle_blood_intra), axis=0)
        shuffle_relative_matrix = np.concatenate((shuffle_relative_inner, shuffle_relative_intra), axis=0)

        shuffle_blood_matrix = torch.tensor(shuffle_blood_matrix, dtype=torch.float32)
        shuffle_relative_matrix = torch.tensor(shuffle_relative_matrix, dtype=torch.float32)


        shuffle_bloodList.append(shuffle_blood_matrix)
        shuffle_relativeList.append(shuffle_relative_matrix)

        shuffle_label = torch.tensor(shuffle_label, dtype=torch.float32)
        shuffle_labelList.append(shuffle_label)

    return shuffle_bloodList, shuffle_relativeList, shuffle_labelList

