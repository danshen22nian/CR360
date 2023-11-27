import os

from utils import get_predInfo
from metric.Metric_SOR import ComputeSOR
from metric.Metric_SA_SOR import ComputeSA_SOR
from metric.NEW_MetricTT import two_twoRelation
import math


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

def convertToNumberOrder(rank_NameOrder, rank_ValueOrder, img_allCats):

    temp_NumRank = []
    for i in range(0, len(rank_NameOrder)):
        temp_NumRank.append(0)

    set_ValueList = rank_ValueOrder.copy()
    set_ValueList = sorted(list(set(set_ValueList)), reverse=True)
    for value_index in range(0, len(set_ValueList)):
        cur_value = set_ValueList[value_index]
        cur_rank = value_index + 1
        for cat_index in range(0, len(rank_ValueOrder)):
            if rank_ValueOrder[cat_index] == cur_value:
                temp_NumRank[cat_index] = cur_rank

    final_numRank = []
    for i in range(0, len(img_allCats)):
        final_numRank.append(0)

    for cat_index in range(0, len(img_allCats)):
        if img_allCats[cat_index] in rank_NameOrder:
            get_index = rank_NameOrder.index(img_allCats[cat_index])
            cat_rank = temp_NumRank[get_index]

            final_numRank[cat_index] = cat_rank

    return final_numRank

def COCO_filter(nameOrder, valueOrder):
    new_nameOrder, new_vlaueOrder = [], []

    for i in range(0, len(nameOrder)):
        if nameOrder[i] in COCO_cat:
            new_nameOrder.append(nameOrder[i])
            new_vlaueOrder.append(valueOrder[i])
    return new_nameOrder, new_vlaueOrder


def GTCat_filter(nameOrder, valueOrder, img_allCats):
    new_nameOrder, new_vlaueOrder = [], []
    for i in range(0, len(nameOrder)):
        if nameOrder[i] in img_allCats:
            new_nameOrder.append(nameOrder[i])
            new_vlaueOrder.append(valueOrder[i])

    return new_nameOrder, new_vlaueOrder

def NonValue_filter(nameOrder, valueOrder):
    new_nameOrder, newValueOrder = [], []
    for i in range(0, len(valueOrder)):
        if math.isnan(valueOrder[i]) == False:
            new_nameOrder.append(nameOrder[i])
            newValueOrder.append(valueOrder[i])

    return new_nameOrder, newValueOrder


def findCommonTT(Pred_nameOrder, Pred_valueOrder, GT_nameOrder, GT_valueOrder):

    pred_rel_list = two_twoRelation(Pred_nameOrder, Pred_valueOrder)
    GT_rel_list = two_twoRelation(GT_nameOrder, GT_valueOrder)


    cur_imgGTAllRel = len(GT_rel_list)
    commenRel_num = 0
    pred_correct_num = 0
    for rel_index in range(0, len(GT_rel_list)):

        cur_GTrel = GT_rel_list[rel_index]

        if cur_GTrel in pred_rel_list:
            pred_correct_num = pred_correct_num + 1
            commenRel_num = commenRel_num + 1

        else:

            reverse_rel = [cur_GTrel[1], cur_GTrel[0]]

            if reverse_rel in pred_rel_list:
                commenRel_num = commenRel_num + 1



    return cur_imgGTAllRel, commenRel_num, pred_correct_num



def ALL_metricCompute(GT_pathList, Pred_NameList, Pred_ValueList):

    only_COCO = True

    GT_splitRelNum = 0

    PredAndGT_commonNum = 0

    Pred_CorrectNum = 0




    sor_list = []
    sa_sor_List = []


    top_1_total = 0
    top_1_check = 0

    for img_index in range(0, len(GT_pathList)):
        img_GTPath = GT_pathList[img_index]

        GT_nameOrder, GT_valueOrder = get_predInfo(img_GTPath)
        Pred_nameOrder, Pred_valueOrder = Pred_NameList[img_index], Pred_ValueList[img_index]

        if only_COCO == True:
            GT_nameOrder, GT_valueOrder = COCO_filter(GT_nameOrder, GT_valueOrder)
            Pred_nameOrder, Pred_valueOrder = COCO_filter(Pred_nameOrder, Pred_valueOrder)

        img_allCats = GT_nameOrder.copy()
        img_allCats.sort()
        Pred_nameOrder, Pred_valueOrder = GTCat_filter(Pred_nameOrder, Pred_valueOrder, img_allCats)

        Pred_nameOrder, Pred_valueOrder = NonValue_filter(Pred_nameOrder, Pred_valueOrder)

        GT_nameOrder, GT_valueOrder = NonValue_filter(GT_nameOrder, GT_valueOrder)

        if len(img_allCats) <= 1:
            continue

        if len(Pred_nameOrder) <= 1:
            GT_splitRelNum = GT_splitRelNum + len(two_twoRelation(GT_nameOrder, GT_valueOrder))
        else:
            cur_imgGTAllRel, cur_commenRel_num, cur_pred_correct_num = findCommonTT(Pred_nameOrder, Pred_valueOrder,
                                                                                    GT_nameOrder, GT_valueOrder)

            GT_splitRelNum = GT_splitRelNum + cur_imgGTAllRel

            PredAndGT_commonNum = PredAndGT_commonNum + cur_commenRel_num

            Pred_CorrectNum = Pred_CorrectNum + cur_pred_correct_num

            GT_numRank = convertToNumberOrder(GT_nameOrder, GT_valueOrder, img_allCats)
            Pred_numRank = convertToNumberOrder(Pred_nameOrder, Pred_valueOrder, img_allCats)

            sor_value = ComputeSOR(GT_numRank, Pred_numRank)
            sa_sor_value = ComputeSA_SOR(GT_numRank, Pred_numRank)


            top_1_total = top_1_total + 1

            if GT_nameOrder[0] == Pred_nameOrder[0]:
                top_1_check = top_1_check + 1



        if len(Pred_nameOrder) <= 1:
            sor_list.append(0)
            sa_sor_List.append(0)
        else:
            sor_list.append(sor_value[0])
            sa_sor_List.append(sa_sor_value)

    sor_sum = 0
    sor_VAL_COUNT = 0
    for i in range(0, len(sor_list)):
        if math.isnan(sor_list[i]) == False:
            sor_sum = sor_sum + float(sor_list[i])
            sor_VAL_COUNT = sor_VAL_COUNT + 1

    sa_sor_sum = 0
    sa_sor_VAL_COUNT = 0
    for i in range(0, len(sa_sor_List)):
        if math.isnan(sa_sor_List[i]) == False:
            sa_sor_sum = sa_sor_sum + float(sa_sor_List[i])
            sa_sor_VAL_COUNT = sa_sor_VAL_COUNT + 1

    tt_s = Pred_CorrectNum / GT_splitRelNum
    tt = Pred_CorrectNum / PredAndGT_commonNum

    top_1_acc = top_1_check / top_1_total


    str_1 = "SA-SOR  : {} , val \ all :{} \ {}".format(sa_sor_sum / len(sa_sor_List), sa_sor_VAL_COUNT, sa_sor_VAL_COUNT)
    str_2 = "SOR : {} , val \ all :{} \ {}".format(sor_sum / len(sor_list), sor_VAL_COUNT, sa_sor_VAL_COUNT)
    str_3 = "TT-S: {} , Pred_CorrectNum \ GT_splitRelNum : {} \ {}".format(tt_s, Pred_CorrectNum, GT_splitRelNum)
    str_4 = "TT: {} , Pred_CorrectNum \ PredAndGT_commonNum : {} \ {}".format(tt, Pred_CorrectNum, PredAndGT_commonNum)
    str_5 = "top_1_acc: {} , top_1_check \ top_1_total : {} \ {}".format(top_1_acc, top_1_check, top_1_total)
    str_list = [str_1, str_2, str_3, str_4, str_5]


    return sor_sum / len(sor_list), tt_s, top_1_acc, str_list



def get_predInfo2(path):

    labelInfo = open(path, "r", encoding="UTF-8")

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




def ALL_metricCompute222(GT_pathList, Pred_NameList, Pred_ValueList, use_DETR, DETR_catPath):

    only_COCO = True

    GT_splitRelNum = 0

    PredAndGT_commonNum = 0

    Pred_CorrectNum = 0


    sor_list = []
    sa_sor_List = []


    top_1_total = 0
    top_1_check = 0

    use_DETR = True

    for img_index in range(0, len(GT_pathList)):
        img_GTPath = GT_pathList[img_index]

        GT_nameOrder, GT_valueOrder = get_predInfo(img_GTPath)
        Pred_nameOrder, Pred_valueOrder = Pred_NameList[img_index], Pred_ValueList[img_index]



        img_name = GT_pathList[img_index].split("\\")[-1]

        cur_DETR = os.path.join(DETR_catPath, img_name)
        DTER_cats, score = get_predInfo2(cur_DETR)


        if only_COCO == True:
            GT_nameOrder, GT_valueOrder = COCO_filter(GT_nameOrder, GT_valueOrder)
            Pred_nameOrder, Pred_valueOrder = COCO_filter(Pred_nameOrder, Pred_valueOrder)

        img_allCats = GT_nameOrder.copy()
        img_allCats.sort()
        Pred_nameOrder, Pred_valueOrder = GTCat_filter(Pred_nameOrder, Pred_valueOrder, img_allCats)

        if use_DETR == True:
            Pred_nameOrder, Pred_valueOrder = GTCat_filter(Pred_nameOrder, Pred_valueOrder, DTER_cats)

        Pred_nameOrder, Pred_valueOrder = NonValue_filter(Pred_nameOrder, Pred_valueOrder)

        GT_nameOrder, GT_valueOrder = NonValue_filter(GT_nameOrder, GT_valueOrder)

        if len(img_allCats) <= 1:
            continue

        if len(Pred_nameOrder) <= 1:
            GT_splitRelNum = GT_splitRelNum + len(two_twoRelation(GT_nameOrder, GT_valueOrder))
        else:
            cur_imgGTAllRel, cur_commenRel_num, cur_pred_correct_num = findCommonTT(Pred_nameOrder, Pred_valueOrder,
                                                                                    GT_nameOrder, GT_valueOrder)

            GT_splitRelNum = GT_splitRelNum + cur_imgGTAllRel

            PredAndGT_commonNum = PredAndGT_commonNum + cur_commenRel_num

            Pred_CorrectNum = Pred_CorrectNum + cur_pred_correct_num

            GT_numRank = convertToNumberOrder(GT_nameOrder, GT_valueOrder, img_allCats)
            Pred_numRank = convertToNumberOrder(Pred_nameOrder, Pred_valueOrder, img_allCats)

            sor_value = ComputeSOR(GT_numRank, Pred_numRank)
            sa_sor_value = ComputeSA_SOR(GT_numRank, Pred_numRank)

            top_1_total = top_1_total + 1

            if GT_nameOrder[0] == Pred_nameOrder[0]:
                top_1_check = top_1_check + 1

        if len(Pred_nameOrder) <= 1:
            sor_list.append(0)
            sa_sor_List.append(0)
        else:
            sor_list.append(sor_value[0])
            sa_sor_List.append(sa_sor_value)

    sor_sum = 0
    sor_VAL_COUNT = 0
    for i in range(0, len(sor_list)):
        if math.isnan(sor_list[i]) == False:
            sor_sum = sor_sum + float(sor_list[i])
            sor_VAL_COUNT = sor_VAL_COUNT + 1

    sa_sor_sum = 0
    sa_sor_VAL_COUNT = 0
    for i in range(0, len(sa_sor_List)):
        if math.isnan(sa_sor_List[i]) == False:
            sa_sor_sum = sa_sor_sum + float(sa_sor_List[i])
            sa_sor_VAL_COUNT = sa_sor_VAL_COUNT + 1

    tt_s = Pred_CorrectNum / GT_splitRelNum
    tt = Pred_CorrectNum / PredAndGT_commonNum

    top_1_acc = top_1_check / top_1_total


    str_1 = "SA-SOR  : {} , val \ all :{} \ {}".format(sa_sor_sum / len(sa_sor_List), sa_sor_VAL_COUNT, sa_sor_VAL_COUNT)
    str_2 = "SOR : {} , val \ all :{} \ {}".format(sor_sum / len(sor_list), sor_VAL_COUNT, sa_sor_VAL_COUNT)
    str_3 = "TT-S: {} , Pred_CorrectNum \ GT_splitRelNum : {} \ {}".format(tt_s, Pred_CorrectNum, GT_splitRelNum)
    str_4 = "TT: {} , Pred_CorrectNum \ PredAndGT_commonNum : {} \ {}".format(tt, Pred_CorrectNum, PredAndGT_commonNum)
    str_5 = "top_1_acc: {} , top_1_check \ top_1_total : {} \ {}".format(top_1_acc, top_1_check, top_1_total)
    str_list = [str_1, str_2, str_3, str_4, str_5]


    return sor_sum / len(sor_list), tt_s, top_1_acc, str_list