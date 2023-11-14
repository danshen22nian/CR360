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

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


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



def read_split_data3(main_train, val, GT):

    all_ways = os.listdir(main_train)
    train_npy = []
    train_txt = []
    for i in range(0, len(all_ways)):
        train = os.path.join(main_train, all_ways[i], "360Cat_NewDataset", "whole_testResults")

        # train blood_inner, blood_intra
        train_blood_inner = os.path.join(train, "blood", "inner_npy")
        train_blood_intra = os.path.join(train, "blood", "intra_npy")
        train_relative_inner = os.path.join(train, "relative", "inner_npy")
        train_relative_intra = os.path.join(train, "relative", "intra_npy")
        # train_npy = []
        # train_txt = []
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


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num

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

# # 自己的预测结果
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

# 将最后 Output 的结果转化为
def outPut_rankLabel(output, img_GTPath):
    # print(output)
    output = output.tolist()
    GT_allCats, prior_score = get_predInfo(img_GTPath)

    cat_toValue = []
    # 把 GT_allCats 对应的 Cat 位置拿出来
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

    # print(Pred_NameOrder)
    # print(Pred_ValueOrder)
#     print("{}".format())
    return Pred_NameOrder, Pred_ValueOrder


def ouputINFO(list):
    print("================================================")
    for i in range(0, len(list)):
        print(list[i])
    print("================================================")


# 保存预测的结果，写入txt
def writeInInfo(save_predResult, img_name, Pred_NameList, Pred_ValueList):
    save_path = os.path.join(save_predResult, img_name + ".txt")
    str_hwole = "finalOrder,"
    for i in range(0, len(Pred_NameList)):
        # line_str = f"{Pred_NameList[i]}, PriorScore:{Pred_ValueList[i]}\n"
        line_str = f"{Pred_NameList[i]}, PriorScore:{str(round(Pred_ValueList[i], 2))}\n"
        f = open(save_path, 'a')
        f.write(line_str)
        f.close()

        if i == len(Pred_NameList)-1:
            #str_hwole = str_hwole + Pred_NameList[i] + ":" + str(Pred_ValueList[i])
            str_hwole = str_hwole + Pred_NameList[i] + ":" + str(round(Pred_ValueList[i], 2))
        else:
            #str_hwole = str_hwole + Pred_NameList[i] + ":" + str(Pred_ValueList[i]) + ","
            str_hwole = str_hwole + Pred_NameList[i] + ":" + str(round(Pred_ValueList[i], 2)) + ","

    Pred_dict = dict(zip(Pred_NameList, Pred_ValueList))
    # Pred_dict = sorted(Pred_dict.items(), key=lambda item: item[1], reverse=True)
    # str = f"FinalOrder!{Pred_dict}\n"
    # f = open(save_path, 'a')
    # f.write(str)
    # f.close()

    f = open(save_path, 'a')

    #
    # final_orderStr = f"FinalOrder!{final_dict}"
    f.write(str_hwole)
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
    # 80 * 80 维的向量
    blood_inner = np.load(cur_npyPath[0][0])
    blood_intra = np.load(cur_npyPath[0][1])
    relative_inner = np.load(cur_npyPath[1][0])
    relative_intra = np.load(cur_npyPath[1][1])
    # result = np.concatenate((array1, array2), axis=0)
    # print(blood_inner.shape)
    # print(relative_inner.shape)
    blood_inner = blood_inner.reshape((1, 80, 80))
    blood_intra = blood_intra.reshape((1, 80, 80))
    relative_inner = relative_inner.reshape((1, 80, 80))
    relative_intra = relative_intra.reshape((1, 80, 80))
    # print(blood_inner.shape)
    # print(relative_inner.shape)
    # print("{}".format())
    blood_matrix = np.concatenate((blood_inner, blood_intra), axis=0)
    relative_matrix = np.concatenate((relative_inner, relative_intra), axis=0)

    return blood_matrix, relative_matrix

def matrix_augment(cur_npyPath, ori_label, ori_Catlist, times):
    # 80 * 80 维的向量
    blood_inner = np.load(cur_npyPath[0][0])
    blood_intra = np.load(cur_npyPath[0][1])
    relative_inner = np.load(cur_npyPath[1][0])
    relative_intra = np.load(cur_npyPath[1][1])

    # blood_inner = blood_inner.reshape((1, 80, 80))
    # blood_intra = blood_intra.reshape((1, 80, 80))
    # relative_inner = relative_inner.reshape((1, 80, 80))
    # relative_intra = relative_intra.reshape((1, 80, 80))

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
        shuffle_label = np.zeros_like(ori_label)  # 对应 shuffle_list 的标签
        # ori_label  # 原来 COCO_catList 下对应的数字排序
        # ori_CatList # 原来 COCO_cat 的list
        for i in range(0, len(ori_Catlist)):
            # 当前处理的类别在 shuffle_list 中的索引
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

        # shuffle_bloodList, shuffle_relativeList, shuffle_labelList
        shuffle_bloodList.append(shuffle_blood_matrix)
        shuffle_relativeList.append(shuffle_relative_matrix)

        shuffle_label = torch.tensor(shuffle_label, dtype=torch.float32)
        shuffle_labelList.append(shuffle_label)

    return shuffle_bloodList, shuffle_relativeList, shuffle_labelList

def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model