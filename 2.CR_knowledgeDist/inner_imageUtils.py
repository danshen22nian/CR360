import math
import os
import numpy as np
from commen_utils import *


rel_weight_FI = 0.2
def inner_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, inner_knowledge_matrix, Bp_insVecPath, BP_imgVecPath):

    # save_sim = os.path.join(save_innerPath, "sim_file")
    # save_catVecMultipy = os.path.join(save_innerPath, "ins_vecNpy")
    # save_score = os.path.join(save_innerPath, "score")

    # if not os.path.exists(save_sim):  # 判在文件夹如果不存在则创建为文件夹
    #     os.makedirs(save_sim)

    if len(blood_SimImgPath) == len(blood_SimImgSimList) == len(blood_SimPseudoInfo):
        for index in range(0, len(blood_SimPseudoInfo)):
            print("{} / {}".format(index, len(blood_SimPseudoInfo)))
            # 读当前相似图片的实例级排序
            cur_pro_rankFilePath = blood_SimPseudoInfo[index]
            ins_nameList, prior_score = readInsOrder(cur_pro_rankFilePath)

            cur_imgPath = blood_SimImgPath[index]
            cur_imgDatasetName = cur_imgPath.split("\\")[-3]
            cur_imgVideo = cur_imgPath.split("\\")[-2]
            cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]


            # cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]

            # print(ins_nameList)
            # print(prior_score)

            # 实例级别的两两排序
            # ins_nameList 带顺序， tt_rel_list 带顺序
            tt_rel_list = ins_Level_TTRelation(ins_nameList, prior_score)
            # print(tt_rel_list)

            # 当前图片的内部每一个实例的语义向量
            # cur_pro_insVecPath = blood_SimSplitVec[index]
            cur_pro_insVecPath = os.path.join(Bp_insVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")
            cur_pro_insVec = np.load(cur_pro_insVecPath)

            # 图片间的第一层置信度
            cur_pro_rankConf = blood_SimImgSimList[index]


            sum_score = sum(prior_score)
            # 如全为0， 直接跳出
            flag = False
            for i in range(0, len(prior_score)):
                if prior_score[i] != 0:
                    flag = True

            # 这张图片中不只一个实例，优先级分数不是全为0
            if len(ins_nameList) > 1 and flag == True:
                # save_imgCatNpy = os.path.join(save_catVecMultipy, cur_imgName)
                # save_imgInsScore = os.path.join(save_score, cur_imgName)
                # if not os.path.exists(save_catVecMultipy):  # 判在文件夹如果不存在则创建为文件夹
                #     os.makedirs(save_catVecMultipy)
                # if not os.path.exists(save_score):  # 判在文件夹如果不存在则创建为文件夹
                #     os.makedirs(save_score)

                # yuan
                # formatted_num = "{:.2f}".format(cur_pro_rankConf)
                # save_imgPath = os.path.join(save_innerPath, cur_imgName + "_" + formatted_num)
                # if not os.path.exists(save_imgPath):  # 判在文件夹如果不存在则创建为文件夹
                #     os.makedirs(save_imgPath)
                # save_scoreTXT = os.path.join(save_imgPath, "score.txt")
                # save_VecNpy = os.path.join(save_imgPath, "VecNpy")
                # if not os.path.exists(save_VecNpy):  # 判在文件夹如果不存在则创建为文件夹
                #     os.makedirs(save_VecNpy)

                for rel_index in range(0, len(tt_rel_list)):
                    # 实例级别的关系矩阵填充
                    cur_rel = tt_rel_list[rel_index]
                    # 拿到当前两个实例关系的前后关系大小, 以及两个分数
                    former_insName = cur_rel[0]
                    last_insName = cur_rel[1]
                    former_indexInScore = ins_nameList.index(former_insName)
                    last_indexInScore = ins_nameList.index(last_insName)
                    former_score = prior_score[former_indexInScore]
                    last_score = prior_score[last_indexInScore]



                    # print(former_indexInScore)
                    # print(last_indexInScore)
                    # print(former_score)
                    # print(last_score)
                    # print("{}".format())

                    # rel_Weight = abs(former_score - last_score) / sum_score * 0.1
                    rel_Weight = abs(former_score - last_score) / sum_score
                    # rel_Weight = math.exp(abs(former_score - last_score) / sum_score * rel_weight_FI)
                    # rel_Weight = np.exp(abs(former_score - last_score) / sum_score * rel_weight_FI)

                    # rel_Weight = np.exp(abs(former_score - last_score) / sum_score * rel_weight_FI)

                    # 关系信息
                    # txt_info = f"{former_insName}:{former_score},{last_insName}:{last_score},diff:{rel_Weight}\n"
                    # f = open(save_scoreTXT, 'a')
                    # f.write(txt_info)
                    # f.close()
                    # save_curRelVecNpy = os.path.join(save_VecNpy, former_insName + "_" + last_insName + ".npy")

                    # COCO_cat
                    former_indexInVec = int(former_insName.split("_")[1])
                    last_iddexInVec = int(last_insName.split("_")[1])
                    former_Vec = cur_pro_insVec[former_indexInVec]
                    last_Vec = cur_pro_insVec[last_iddexInVec]

                    # print(former_Vec.shape)
                    # print(last_Vec.shape)
                    # 前大后小没问题
                    # 大
                    former_Vec = former_Vec.reshape((80, 1))
                    # 小
                    last_Vec = last_Vec.reshape((1, 80))
                    cur_img_KnowMatrix = np.dot(former_Vec, last_Vec)

                    # cur_img_KnowMatrix = np.exp(cur_img_KnowMatrix)
                    # 保存概率矩阵
                    # np.save(save_curRelVecNpy, cur_img_KnowMatrix)

                    cur_img_KnowMatrix = cur_img_KnowMatrix * cur_pro_rankConf * rel_Weight
                    # cur_img_KnowMatrix = cur_img_KnowMatrix * cur_pro_rankConf * 1

                    inner_knowledge_matrix = cur_img_KnowMatrix + inner_knowledge_matrix
                    # 将对角线元素设置为0
                    np.fill_diagonal(inner_knowledge_matrix, 0)

                    # print("{}".format())
                    # # COCO_catList
                    # for out_index in range(0, len(COCO_cat)):
                    #     out_poss = former_Vec[out_index]
                    #     for in_index in range(0, len(COCO_cat)):
                    #         in_poss = last_Vec[in_index]
                    #         if out_index != in_index:
                    #             inner_knowledge_matrix[out_index][in_index] = inner_knowledge_matrix[out_index][in_index] + cur_pro_rankConf * rel_Weight * out_poss * in_poss

  #   print(inner_knowledge_matrix)
#     print("{}".format())
    return inner_knowledge_matrix

                    # Vec 中的可能性和 对应 COCO_cat


                    # if cur_rel in cur_img_Rels:
                    #     former_object = cur_rel[0]
                    #     last_object = cur_rel[1]
                    #     # 更新矩阵的x,y
                    #     formerIndex_inNumpy = img_AllObjects.index(former_object)
                    #     lastIndex_inNumpy = img_AllObjects.index(last_object)
                    #
                    #     # 拿到两个物体的prior score
                    #     former_score_index = object_name.index(former_object)
                    #     last_score_index = object_name.index(last_object)
                    #     former_score = prior_score[former_score_index]
                    #     last_score = prior_score[last_score_index]
                    #     rel_weight = rel_weight_compute(former_score, last_score, sum_score)
                    #     # print("关系权重：{} {} ： {}".format(former_object, last_object, rel_weight))
                    #     if former_score > last_score:
                    #         orderNumpy[formerIndex_inNumpy][lastIndex_inNumpy] = 1 * rel_weight * cur_pro_rankConf + orderNumpy[formerIndex_inNumpy][lastIndex_inNumpy]
                    #
                    #     # 设定下三角矩阵为 x 《 y 的关系，即所有numpy[y][x]的关系
                    #     elif last_score > former_score:
                    #         orderNumpy[lastIndex_inNumpy][formerIndex_inNumpy] = 1 * rel_weight * cur_pro_rankConf + orderNumpy[lastIndex_inNumpy][formerIndex_inNumpy]

            # row_sum = np.sum(orderNumpy, axis=1)
            # # 这里的列num指的是 想 x < y的次数
            # column_sum = np.sum(orderNumpy, axis=0)
            # total_num = row_sum + column_sum
            # final_weights = row_sum / total_num
            # final_weights = np.round(final_weights, decimals=2)
            # for i in range(0, len(final_weights)):
            #     if isinstance(final_weights[i], (int, float)) == False:
            #         print("{}".format())
            # # print(final_weights)
            # finalDict = dict(zip(img_AllObjects, final_weights))
            # finalOrder = sorted(finalDict.items(), key=lambda x: x[1], reverse=True)
            #
            # final_nameOrder = []
            # final_score = []
            #
            # if len(finalOrder) == 1:
            #     cur_name, cur_weight = finalOrder[i]
            #     final_nameOrder.append(cur_name)
            #     final_score.append(float(1))
            # else:
            #     for i in range(0, len(finalOrder)):
            #         cur_name, cur_weight = finalOrder[i]
            #         final_nameOrder.append(cur_name)
            #         final_score.append(cur_weight)



            # print("最终关系矩阵{}".format(orderNumpy))


