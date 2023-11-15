import os

from commen_utils import *
from inner_imageUtils import *
from intra_imageUtils import *

def blood_related_KnowlodgeDistill(blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath, Bp_insVecPath, BP_imgVecPath):

    inner_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))
    intra_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))

    blood_SimPseudoInfo = []

    for i in range(0, len(blood_SimImgPath)):
        cur_imgPath = blood_SimImgPath[i]
        cur_imgDatasetName = cur_imgPath.split("\\")[-3]
        cur_imgVideo = cur_imgPath.split("\\")[-2]
        cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]

        cur_img_PseudoRank = os.path.join(BP_PseudoPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")

        blood_SimPseudoInfo.append(cur_img_PseudoRank)


    inner_knowledge_matrix = inner_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, inner_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)
    intra_knowledge_matrix = intra_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, intra_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)
    expert_knowledge_matrix = combine_InnerAndIntra(inner_knowledge_matrix, intra_knowledge_matrix)

    return inner_knowledge_matrix, intra_knowledge_matrix, expert_knowledge_matrix


def relative_related_knowledgeDistill(blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath,
                                    Bp_insVecPath, BP_imgVecPath, used_Family_imgPath, set_secondSearch_per,
                                bp_npy, bp_All_ImgPath):

    rel_numSet = 500 - len(blood_SimImgPath)
    relative_familiesPath = []
    relative_simList = []
    for blood_index in range(0, len(blood_SimImgPath)):
        cur_blood_path = blood_SimImgPath[blood_index]
        cur_imgDatasetName = cur_blood_path.split("\\")[-3]
        cur_imgVideo = cur_blood_path.split("\\")[-2]
        cur_imgName = cur_blood_path.split("\\")[-1].split(".")[0]


        cur_blood_vec_path = os.path.join(BP_imgVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")
        cur_vec = np.load(cur_blood_vec_path)

        sim_numpy = cosine_similarity(cur_vec, bp_npy)
        sim_numpy = normalize_matrix(sim_numpy)
        if set_secondSearch_per != None:
            most_sim_num = int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))
            most_sim_index = find_top_n_indices(sim_numpy, most_sim_num)
        else:
            most_sim_index = find_top_n_indices(sim_numpy, len(sim_numpy))
        sim_list = list(most_sim_index)
        for i in range(0, len(sim_list)):

            if bp_All_ImgPath[sim_list[i]] not in used_Family_imgPath:

                if len(relative_familiesPath) < rel_numSet:

                    relative_familiesPath.append(bp_All_ImgPath[sim_list[i]])
                    relative_simList.append(sim_numpy[sim_list[i]])
                    used_Family_imgPath.append(bp_All_ImgPath[sim_list[i]])
                else:

                    min_index = relative_simList.index(min(relative_simList))
                    relative_simList[min_index] = sim_numpy[sim_list[i]]
                    relative_familiesPath[min_index] = bp_All_ImgPath[sim_list[i]]
                    used_Family_imgPath.append(bp_All_ImgPath[sim_list[i]])

    print("relative_simNumber:{}".format(len(relative_familiesPath)))
    relative_SimPseudoInfo = []

    for i in range(0, len(relative_familiesPath)):
        cur_imgPath = relative_familiesPath[i]
        cur_imgDatasetName = cur_imgPath.split("\\")[-3]
        cur_imgVideo = cur_imgPath.split("\\")[-2]
        cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]

        cur_img_PseudoRank = os.path.join(BP_PseudoPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")


        relative_SimPseudoInfo.append(cur_img_PseudoRank)


    inner_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))
    intra_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))


    inner_knowledge_matrix = inner_imageKnowledge(relative_familiesPath, relative_simList, relative_SimPseudoInfo,
                                                  inner_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)
    intra_knowledge_matrix = intra_imageKnowledge(relative_familiesPath, relative_simList, relative_SimPseudoInfo,
                                                  intra_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)

    general_knowledge_matrix = combine_InnerAndIntra(inner_knowledge_matrix, intra_knowledge_matrix)

    return inner_knowledge_matrix, intra_knowledge_matrix, general_knowledge_matrix


