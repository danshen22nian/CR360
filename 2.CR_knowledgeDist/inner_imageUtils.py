import math
import os
import numpy as np
from commen_utils import *



def inner_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, inner_knowledge_matrix, Bp_insVecPath, BP_imgVecPath):


    if len(blood_SimImgPath) == len(blood_SimImgSimList) == len(blood_SimPseudoInfo):
        for index in range(0, len(blood_SimPseudoInfo)):
            print("{} / {}".format(index, len(blood_SimPseudoInfo)))

            cur_pro_rankFilePath = blood_SimPseudoInfo[index]
            ins_nameList, prior_score = readInsOrder(cur_pro_rankFilePath)

            cur_imgPath = blood_SimImgPath[index]
            cur_imgDatasetName = cur_imgPath.split("\\")[-3]
            cur_imgVideo = cur_imgPath.split("\\")[-2]
            cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]

            tt_rel_list = ins_Level_TTRelation(ins_nameList, prior_score)

            cur_pro_insVecPath = os.path.join(Bp_insVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")
            cur_pro_insVec = np.load(cur_pro_insVecPath)

            cur_pro_rankConf = blood_SimImgSimList[index]


            sum_score = sum(prior_score)

            flag = False
            for i in range(0, len(prior_score)):
                if prior_score[i] != 0:
                    flag = True

            if len(ins_nameList) > 1 and flag == True:

                for rel_index in range(0, len(tt_rel_list)):

                    cur_rel = tt_rel_list[rel_index]

                    former_insName = cur_rel[0]
                    last_insName = cur_rel[1]
                    former_indexInScore = ins_nameList.index(former_insName)
                    last_indexInScore = ins_nameList.index(last_insName)
                    former_score = prior_score[former_indexInScore]
                    last_score = prior_score[last_indexInScore]

                    rel_Weight = abs(former_score - last_score) / sum_score


                    former_indexInVec = int(former_insName.split("_")[1])
                    last_iddexInVec = int(last_insName.split("_")[1])
                    former_Vec = cur_pro_insVec[former_indexInVec]
                    last_Vec = cur_pro_insVec[last_iddexInVec]


                    former_Vec = former_Vec.reshape((80, 1))
                    last_Vec = last_Vec.reshape((1, 80))
                    cur_img_KnowMatrix = np.dot(former_Vec, last_Vec)


                    cur_img_KnowMatrix = cur_img_KnowMatrix * cur_pro_rankConf * rel_Weight
                    inner_knowledge_matrix = cur_img_KnowMatrix + inner_knowledge_matrix

                    np.fill_diagonal(inner_knowledge_matrix, 0)

    return inner_knowledge_matrix

