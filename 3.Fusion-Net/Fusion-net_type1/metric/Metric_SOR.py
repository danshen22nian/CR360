import os
import scipy.stats as sc
import pickle
import numpy as np

def extract_spr_value(data_list):
    use_idx_list = []
    spr = []
    for i in range(len(data_list)):
        s = data_list[i][1]

        if s == 1:
            spr.append(s)
            use_idx_list.append(i)
        elif s and not np.isnan(s[0]):
            spr.append(s[0])
            use_idx_list.append(i)

    return spr, use_idx_list


def cal_avg_spr(data_list):
    spr = np.array(data_list)
    avg = np.average(spr)
    return avg


def get_norm_spr(spr_value):
    r_min = -1
    r_max = 1

    norm_spr = (spr_value - r_min) / (r_max - r_min)

    return norm_spr

def eval_spr(spr_data_path):
    with open(spr_data_path, "rb") as f:
        spr_all_data = pickle.load(f)

    spr_data, spr_use_idx = extract_spr_value(spr_all_data)

    pos_l = []
    neg_l = []
    for i in range(len(spr_data)):
        if spr_data[i] > 0:
            pos_l.append(spr_data[i])
        else:
            neg_l.append(spr_data[i])

    print("Positive SPR: ", pos_l)
    print("Negative SPR: ", neg_l)
    print("Positive SPR: ", len(pos_l))
    print("Negative SPR: ", len(neg_l))

    avg_spr = cal_avg_spr(spr_data)
    avg_spr_norm = get_norm_spr(avg_spr) #利用[-1,1]进行归一化

    print("\n----------------------------------------------------------")
    print("Data path: ", spr_data_path)
    print(len(spr_data), "/", len(spr_all_data), " - ", (len(spr_all_data) - len(spr_data)), "Images Not used")
    print("Average SPR Saliency: ", avg_spr)
    print("Average SPR Saliency Normalized: ", avg_spr_norm)

    return pos_l, neg_l, spr_data, spr_all_data, avg_spr, avg_spr_norm


def get_usable_salient_objects_agreed(image_1_list, image_2_list):

    rm_list = []
    for idx in range(len(image_1_list)):
        v = image_1_list[idx]
        v2 = image_2_list[idx]

        if v == 0 or v2 == 0:
            rm_list.append(idx)


    use_list = list(range(0, len(image_1_list)))
    use_list = list(np.delete(np.array(use_list), rm_list))


    x = np.array(image_1_list)
    y = np.array(image_2_list)
    x = list(np.delete(x, rm_list))
    y = list(np.delete(y, rm_list))

    return x, y, use_list




def ComputeSOR(GT_numRank, Pred_numRank):
    gt_ranks, pred_Ranks, use_indices_list = \
        get_usable_salient_objects_agreed(GT_numRank, Pred_numRank)
    spr = None

    if len(gt_ranks) > 1:
        spr = sc.spearmanr(gt_ranks, pred_Ranks)
    elif len(gt_ranks) == 1:
        spr = (1,1)

    return spr
