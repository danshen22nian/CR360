import math
import os
import numpy as np
from commen_utils import *




# intra_relThreshold = 0.3
# intra_relThreshold = 0.5
# intra_relThreshold = 0.2
# intra_relThreshold = 0.5
intra_relThreshold = 0.2
# 原来是0.5
# intra_relSimThreshold = 0.5
# 图片之间的相似度
intra_relSimThreshold = 0.7
# intra_relSimThreshold = 0.8


# 两两配对
# def intra_match(blood_SimImgPath, BP_imgVecPath):
#
#     matched_pair = []
#     for i in range(0, len(blood_SimImgPath)):
#         anc_imgPath = blood_SimImgPath[i]
#
#         cur_ancImgDatasetName = anc_imgPath.split("\\")[-3]
#         cur_ancImgVideo = anc_imgPath.split("\\")[-2]
#         cur_ancImgName = anc_imgPath.split("\\")[-1].split(".")[0]
#         anc_imgVec =
#
#         for j in range(i, len(blood_SimImgPath)):
#             com_imgPath = blood_SimImgPath[j]
#             matched_pair = [anc_imgPath, com_imgPath]


def intra_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, intra_knowledge_matrix, Bp_insVecPath, BP_imgVecPath):

    # 找到两张图片之间存在这种关系的物体以及他们之间的
    # print(img_AllObjects)

    for out_index in range(0, len(blood_SimImgPath)):
        print("{} \ {}".format(out_index, len(blood_SimImgPath)))
        cur_anchorImgPath = blood_SimImgPath[out_index]

        # cur_ancImgName = cur_anchorImgPath.split("\\")[-1].split(".")[0]

        # 先读伪标签路径，# 以及该图片内的排序关系
        cur_anchorImg_lebelPath = blood_SimPseudoInfo[out_index]
        anc_ins_name, anc_ins_score = readInsOrder(cur_anchorImg_lebelPath)

        cur_ancImgDatasetName = cur_anchorImgPath.split("\\")[-3]
        cur_ancImgVideo = cur_anchorImgPath.split("\\")[-2]
        cur_ancImgName = cur_anchorImgPath.split("\\")[-1].split(".")[0]

        anc_flag = False
        for i in range(0, len(anc_ins_score)):
            if anc_ins_score[i] != 0:
                anc_flag = True
        if anc_flag == False or len(anc_ins_name) <= 1:
            continue

        # 转化为权重关系
        anc_ins_weight = convertToWeiList(anc_ins_score)

        # anc 图片的split_VEC,还有img_VEC
        cur_ancImgVecPath = os.path.join(BP_imgVecPath, cur_ancImgDatasetName, cur_ancImgVideo, cur_ancImgName + ".npy")
        cur_ancInsVecPath = os.path.join(Bp_insVecPath, cur_ancImgDatasetName, cur_ancImgVideo, cur_ancImgName + ".npy")
        cur_ancImgVec = np.load(cur_ancImgVecPath)
        cur_ancInsVec = np.load(cur_ancInsVecPath)

        for inner_index in range(out_index, len(blood_SimImgPath)):
            cur_comImgPath = blood_SimImgPath[inner_index]
            # cur_comImgName = cur_comImgPath.split("\\")[-1].split(".")[0]
            cur_comImgDatasetName = cur_comImgPath.split("\\")[-3]
            cur_comImgVideo = cur_comImgPath.split("\\")[-2]
            cur_comImgName = cur_comImgPath.split("\\")[-1].split(".")[0]

            # 先读伪标签路径，# 以及该图片内的排序关系
            cur_comImg_lebelPath = blood_SimPseudoInfo[inner_index]
            com_ins_name, com_ins_score = readInsOrder(cur_comImg_lebelPath)

            com_flag = False
            for i in range(0, len(com_ins_score)):
                if com_ins_score[i] != 0:
                    com_flag = True
            if com_flag == False or len(com_ins_name) <= 1:
                continue

            # 转化为权重关系
            com_ins_weight = convertToWeiList(com_ins_score)
            # com 图片的split_VEC,还有img_VEC
            cur_comImgVecPath = os.path.join(BP_imgVecPath, cur_comImgDatasetName, cur_comImgVideo, cur_comImgName + ".npy")
            cur_comInsVecPath = os.path.join(Bp_insVecPath, cur_comImgDatasetName, cur_comImgVideo, cur_comImgName + ".npy")
            cur_comImgVec = np.load(cur_comImgVecPath)
            cur_comInsVec = np.load(cur_comInsVecPath)


            # anc_img 和 com_img之间的相似度
            ancAndcom_sim = cosine_similarity_vec(cur_comImgVec, cur_ancImgVec)
            # print(ancAndcom_sim)
            if intra_relSimThreshold != None:
                if ancAndcom_sim < intra_relSimThreshold:
                    continue
            # print(ancAndcom_sim)

            # anc_ins_name和 com_ins_name 的组合
            ancAndcom_matchList = [[a, b] for a in anc_ins_name for b in com_ins_name]

            for rel_index in range(0, len(ancAndcom_matchList)):
                rel_ancInsName = ancAndcom_matchList[rel_index][0]
                rel_comInsName = ancAndcom_matchList[rel_index][1]
                # 当前实例的显著性比重
                anc_insIndexInWeight = anc_ins_name.index(rel_ancInsName)
                ancInsWeight = anc_ins_weight[anc_insIndexInWeight]
                com_insIndexInWeight = com_ins_name.index(rel_comInsName)
                comInsWeight = com_ins_weight[com_insIndexInWeight]

                # 实例在他的insVec中的索引
                ancInsIndexInVec = int(rel_ancInsName.split("_")[1])
                comInsIndexInVec = int(rel_comInsName.split("_")[1])
                # 实例在他的insVec中 向量
                ancInsVec = cur_ancInsVec[ancInsIndexInVec]
                comInsVec = cur_comInsVec[comInsIndexInVec]


                cur_rel_Weight = abs(ancInsWeight - comInsWeight)
                # cur_rel_Weight = math.exp(abs(ancInsWeight - comInsWeight) * rel_weight_FI)
                # cur_rel_Weight = np.exp(abs(ancInsWeight - comInsWeight) * rel_weight_FI)

                if cur_rel_Weight >= intra_relThreshold:
                    # cur_rel_Weight = abs(ancInsWeight - comInsWeight) * 0.1
                    # cur_rel_Weight = abs(ancInsWeight - comInsWeight)
                    # # 保存关系
                    # save_ancPath = os.path.join(save_intraPath, cur_ancImgName)
                    # formatted_num = "{:.2f}".format(ancAndcom_sim)
                    # save_comPath = os.path.join(save_ancPath, cur_comImgName + "_" + formatted_num)
                    # if not os.path.exists(save_comPath):  # 判在文件夹如果不存在则创建为文件夹
                    #     os.makedirs(save_comPath)
                    # save_scoreTXT = os.path.join(save_comPath, "score.txt")
                    # save_VecNpy = os.path.join(save_comPath, "VecNpy")
                    # if not os.path.exists(save_VecNpy):  # 判在文件夹如果不存在则创建为文件夹
                    #     os.makedirs(save_VecNpy)

                    if ancInsWeight > comInsWeight:
                        # 还是让大的vec在前，小的score在后
                        big_insVec = ancInsVec
                        small_insVec = comInsVec

                        # score_TXT = f"{cur_ancImgName}:{rel_ancInsName},{ancInsWeight}!{cur_comImgName}:{rel_comInsName},{comInsWeight}!{cur_rel_Weight}\n"
                        # save_curRelVecNpy = os.path.join(save_VecNpy, rel_ancInsName + "_" + rel_comInsName + ".npy")

                    else:
                        big_insVec = comInsVec
                        small_insVec = ancInsVec

                        # score_TXT = f"{cur_comImgName}:{rel_comInsName},{comInsWeight}!{cur_ancImgName}:{rel_ancInsName},{ancInsWeight}!diff:{cur_rel_Weight}\n"
                        # save_curRelVecNpy = os.path.join(save_VecNpy, rel_comInsName + "_" + rel_ancInsName + ".npy")

                    # f = open(save_scoreTXT, 'a')
                    # f.write(score_TXT)
                    # f.close()


                    # 大的vec搞成80*1 ,
                    big_insVec = big_insVec.reshape((80, 1))
                    small_insVec = small_insVec.reshape((1, 80))
                    cur_img_KnowMatrix = np.dot(big_insVec, small_insVec)

                    # r
                    # cur_img_KnowMatrix = np.exp(cur_img_KnowMatrix)
                    # np.save(save_curRelVecNpy, cur_img_KnowMatrix)

                    # rel 关系的权重， ancAndcom_sim图片之间的相似性
                    cur_img_KnowMatrix = cur_img_KnowMatrix * ancAndcom_sim * cur_rel_Weight

                    intra_knowledge_matrix = intra_knowledge_matrix + cur_img_KnowMatrix
                    np.fill_diagonal(intra_knowledge_matrix, 0)


 #    print(intra_knowledge_matrix)
#    print("{}".format())
    return intra_knowledge_matrix
    #
    #     if anchor_count != 0:
    #         cur_DatasetName = cur_anchorImgPath.split("\\")[-3]
    #         cur_Video = cur_anchorImgPath.split("\\")[-2]
    #         cur_Name = cur_anchorImgPath.split("\\")[-1].split(".")[0]
    #         anchor_vecPath = os.path.join(BP_VecPath, cur_DatasetName, cur_Video, cur_Name + ".npy")
    #         np.fill_diagonal(intra_knowledge_matrix, 0)
    #
    #         # 找到它的vec
    #         anchor_vec = np.load(anchor_vecPath)
    #     # print("anchor_allInfo:")
    #     # print(anc_object_name)
    #     # print(anc_prior_score)
    #     # print("anchor_valInfo:")
    #     # print(anchor_valName)
    #     # print(anchor_NameToScore)
    #     # print("{]".format())
    #
    #
    #     if img_index + 1 != len(cur_imgAllRel) - 1 and anchor_count != 0 and anc_flag != False and len(anchor_valName) != 1:
    #         for inner_index in range(img_index + 1, len(cur_imgAllRel)):
    #             # 同样找到当前图片有无在img_AllObjects出现的物体，如果没有就就跳出，有的话同样拿出俩放到
    #             cur_ComImg_lebelPath = cur_imgAllPsrudoInfo[inner_index]
    #             # print(cur_ComImg_lebelPath)
    #             com_object_name, com_prior_score = readOrderInfo(cur_ComImg_lebelPath)
    #             com_scoreSum = sum(com_prior_score)
    #
    #             if com_scoreSum == 0:
    #                 continue
    #             # 否则找到所有img_ALLobjects 中所出现的类别，以及它占当前这段关系的权重，放到两个列表中
    #             com_valName = []
    #             com_NameToScore = []
    #
    #             com_count = 0
    #             for i in range(0, len(com_object_name)):
    #                 if com_object_name[i] in img_AllObjects:
    #                     com_count = com_count + 1
    #                     com_valName.append(com_object_name[i])
    #                     # anchor_NameToScore.append(prior_score[i])
    #                     com_NameToScore.append(com_prior_score[i] / com_scoreSum)
    #
    #             com_flag = False
    #             for i in range(0, len(com_prior_score)):
    #                 if com_prior_score[i] != 0:
    #                     com_flag = True
    #
    #             if com_count != 0 and com_flag != False and len(com_object_name) != 1:
    #                 cur_DatasetName = cur_ComImg_lebelPath.split("\\")[-3]
    #                 cur_Video = cur_ComImg_lebelPath.split("\\")[-2]
    #                 cur_Name = cur_ComImg_lebelPath.split("\\")[-1].split(".")[0]
    #                 com_vecPath = os.path.join(BP_VecPath, cur_DatasetName, cur_Video, cur_Name + ".npy")
    #
    #                 # 找到它的vec
    #                 com_vec = np.load(com_vecPath)
    #
    #                 # 两段比较的关系的相似度
    #                 cur_com_conf = cosine_similarity_vec(anchor_vec, com_vec)
    #
    #                 # com_valName， com_NameToScore
    #                 # anchor_valName = []， anchor_NameToScore = []
    #                 # 对两张图片中都有的 测试图片中的关系进行组合，然后，再填关系
    #
    #                 # print("com_allInfo:")
    #                 # print(com_object_name)
    #                 # print(com_prior_score)
    #                 # print("com_valInfo:")
    #                 # print(com_valName)
    #                 # print(com_NameToScore)
    #                 # print("{}".format())
    #
    #                 # 想矩阵中添加关系
    #                 for anc_index in range(0, len(anchor_valName)):
    #                     cur_ancName = anchor_valName[anc_index]
    #                     cur_ancScore = anchor_NameToScore[anc_index]
    #                     # com_valName = []
    #                     # com_NameToScore = []
    #                     for com_index in range(0, len(com_valName)):
    #                         cur_comName = com_valName[com_index]
    #                         cur_comScore = com_NameToScore[com_index]
    #
    #
    #                         # 组合两个名字
    #                         if cur_ancName != cur_comName:
    #                             two_twoName = []
    #                             two_twoScore = []
    #
    #                             two_twoName.append(cur_ancName)
    #                             two_twoName.append(cur_comName)
    #                             two_twoScore.append(cur_ancScore)
    #                             two_twoScore.append(cur_comScore)
    #
    #
    #                             cur_rel = []
    #                             for i in range(0, len(two_twoName)):
    #                                 cur_rel.append(two_twoName[i])
    #                             cur_rel.sort()
    #
    #                             rel_weight = intra_rel_weight_compute(two_twoScore[0], two_twoScore[1])
    #
    #                             if cur_rel in cur_img_Rels and rel_weight > intra_relThreshold:
    #                                 former_object = cur_rel[0]
    #                                 last_object = cur_rel[1]
    #                                 # 更新矩阵的x,y
    #                                 formerIndex_inNumpy = img_AllObjects.index(former_object)
    #                                 lastIndex_inNumpy = img_AllObjects.index(last_object)
    #
    #                                 # 拿到两个物体的prior score
    #                                 former_score_index = two_twoName.index(former_object)
    #                                 last_score_index = two_twoName.index(last_object)
    #                                 former_score = two_twoScore[former_score_index]
    #                                 last_score = two_twoScore[last_score_index]
    #                                 # rel_weight = rel_weight_compute(former_score, last_score, sum_score)
    #                                 # rel_weight = inner_rel_weight_compute(former_score, last_score)
    #                                 # print("关系权重：{} {} ： {}".format(former_object, last_object, rel_weight))
    #                                 if former_score > last_score:
    #                                     intra_matrix[formerIndex_inNumpy][
    #                                         lastIndex_inNumpy] = 1 * rel_weight * cur_com_conf + \
    #                                                              intra_matrix[formerIndex_inNumpy][lastIndex_inNumpy]
    #
    #                                 # 设定下三角矩阵为 x 《 y 的关系，即所有numpy[y][x]的关系
    #                                 elif last_score > former_score:
    #                                     intra_matrix[lastIndex_inNumpy][
    #                                         formerIndex_inNumpy] = 1 * rel_weight * cur_com_conf + \
    #                                                                intra_matrix[lastIndex_inNumpy][formerIndex_inNumpy]
    #
    # return intra_matrix
    #
    #
    #
    #
