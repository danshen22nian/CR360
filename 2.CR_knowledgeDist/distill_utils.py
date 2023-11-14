import os

from commen_utils import *
from inner_imageUtils import *
from intra_imageUtils import *







# 不对数据进图片内外间关系的过滤，直接由blood中找到语义相似的信息后对，其中出现的了的，rankFile中的显著性值进行累加 得到测试图片的顺序
def noKnowlegdeDistill(blood_SimImgPath, blood_SimImgSimList, img_AllObjects, BP_mainFile):
    # cur_img_Rels = two_two_Relation(img_AllObjects)
    # orderNumpy = np.zeros((len(img_AllObjects), len(img_AllObjects)))
    BP_PseudoPath = os.path.join(BP_mainFile, "pseudo_label_transal")
    blood_SimPseudoInfo = []
    # 找到与 blood_sim 的所有图片路径，以及 对应的伪标签路径
    for i in range(0, len(blood_SimImgPath)):
        cur_imgPath = blood_SimImgPath[i]
        cur_imgDatasetName = cur_imgPath.split("\\")[-3]
        cur_imgVideo = cur_imgPath.split("\\")[-2]
        cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]

        cur_img_PseudoRank = os.path.join(BP_PseudoPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")
        blood_SimPseudoInfo.append(cur_img_PseudoRank)


    test_imgToPriorScore = []
    for i in range(0, len(img_AllObjects)):
        test_imgToPriorScore.append(0)

    # 直直接将其中所有的 img_AllObjects 有的 目标
    for index in range(0, len(blood_SimPseudoInfo)):
        cur_pro_rankFilePath = blood_SimPseudoInfo[index]
        object_name, prior_score = readOrderInfo(cur_pro_rankFilePath)

        sum_score = sum(prior_score)
        # 保证一个该张图片中出现了2个以上的物体，然后才承认该条数据是有意义的
        count_num = 0
        for i in range(0, len(object_name)):
            if object_name[i] in img_AllObjects:
                count_num = count_num + 1
        if count_num >= 2:
            for obj_index in range(0, len(object_name)):
                cur_objName = object_name[obj_index]
                if cur_objName in img_AllObjects:
                    get_index = img_AllObjects.index(cur_objName)
                    test_imgToPriorScore[get_index] = test_imgToPriorScore[get_index] + float(prior_score[obj_index] / sum_score)


    # 保留两位小数
    for i in range(0, len(test_imgToPriorScore)):
        test_imgToPriorScore[i] = round(test_imgToPriorScore[i], 2)


    return img_AllObjects, test_imgToPriorScore



def blood_related_KnowlodgeDistill(blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath, Bp_insVecPath, BP_imgVecPath):


    # save_innerPath = os.path.join(save_test_bloodPath, "inner_Info")
    # save_intraPath = os.path.join(save_test_bloodPath, "intra_Info")

    # 知识矩阵应该 和 splitVec 直接和 COCO_cat 列表对应起来  还是保证横轴0 》 1 的时候填到matrix[0][1] or matrix[1][0]
    inner_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))
    intra_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))

    blood_SimPseudoInfo = []
    # blood_SimSplitVec = []
    # blood_SimImgVec = []

    # 找到与 blood_sim 的所有图片路径，以及 对应的伪标签路径
    for i in range(0, len(blood_SimImgPath)):
        cur_imgPath = blood_SimImgPath[i]
        cur_imgDatasetName = cur_imgPath.split("\\")[-3]
        cur_imgVideo = cur_imgPath.split("\\")[-2]
        cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]
        # 这里的label 和 Vec 都是图片内实例级别的
        cur_img_PseudoRank = os.path.join(BP_PseudoPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")
        # cur_img_splitVec = os.path.join(Bp_insVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")
        # cur_img_imgVec = os.path.join(BP_imgVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")

        blood_SimPseudoInfo.append(cur_img_PseudoRank)
        # blood_SimSplitVec.append(cur_img_splitVec)
        # blood_SimImgVec.append(cur_img_imgVec)

    inner_knowledge_matrix = inner_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, inner_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)
    intra_knowledge_matrix = intra_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, intra_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)

    # 读取 最终80维 的矩阵
    expert_knowledge_matrix = combine_InnerAndIntra(inner_knowledge_matrix, intra_knowledge_matrix)

    return inner_knowledge_matrix, intra_knowledge_matrix, expert_knowledge_matrix


# Relative_SimThreshold = 0.97
rel_numSet = 328
def relative_related_knowledgeDistill(blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath,
                                    Bp_insVecPath, BP_imgVecPath, used_Family_imgPath, set_secondSearch_per,
                                bp_npy, bp_All_ImgPath):

    # save_innerPath = os.path.join(save_test_relativePath, "inner_Info")
    # save_intraPath = os.path.join(save_test_relativePath, "intra_Info")

    # print(blood_SimImgPath)
    # print(len(blood_SimImgPath))
    # 从blood往下延伸 里面放的一只保证是相似度最高的那一批数据
    relative_familiesPath = []
    relative_simList = []
    for blood_index in range(0, len(blood_SimImgPath)):
        cur_blood_path = blood_SimImgPath[blood_index]
        cur_imgDatasetName = cur_blood_path.split("\\")[-3]
        cur_imgVideo = cur_blood_path.split("\\")[-2]
        cur_imgName = cur_blood_path.split("\\")[-1].split(".")[0]

        # 找到v图片向量路径
        cur_blood_vec_path = os.path.join(BP_imgVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")
        cur_vec = np.load(cur_blood_vec_path)
        # 然后求相似性得到一个表示矩阵
        sim_numpy = cosine_similarity(cur_vec, bp_npy)
        sim_numpy = normalize_matrix(sim_numpy)
        # print(sim_numpy)

        if set_secondSearch_per != None:
            most_sim_num = int(np.floor(len(bp_All_ImgPath) * set_secondSearch_per * 0.01))
            most_sim_index = find_top_n_indices(sim_numpy, most_sim_num)
        else:
            # 参考所有的
            most_sim_index = find_top_n_indices(sim_numpy, len(sim_numpy))

        # print(most_sim_index)
        sim_list = list(most_sim_index)
        # print(sim_list)

        # print(sim_list)
        # print("{}".format())
        for i in range(0, len(sim_list)):
            # print(bp_All_ImgPath[sim_list[i]])
            if bp_All_ImgPath[sim_list[i]] not in used_Family_imgPath:
                # if sim_numpy[sim_list[i]] > Relative_SimThreshold:
                if len(relative_familiesPath) < rel_numSet:
                    # print("-")
                    relative_familiesPath.append(bp_All_ImgPath[sim_list[i]])
                    relative_simList.append(sim_numpy[sim_list[i]])
                    used_Family_imgPath.append(bp_All_ImgPath[sim_list[i]])

                    # print(relative_familiesPath)
                    # print(relative_simList)
                else:
                    # print("-")
                    # 替换掉相似度最小的那个元素的索引
                    min_index = relative_simList.index(min(relative_simList))
                    relative_simList[min_index] = sim_numpy[sim_list[i]]
                    relative_familiesPath[min_index] = bp_All_ImgPath[sim_list[i]]
                    used_Family_imgPath.append(bp_All_ImgPath[sim_list[i]])

    # print(relative_familiesPath)
    # print(relative_simList)
    # print(len(relative_familiesPath))
    # print(len(relative_simList))
    # print("{}".format())


    # print(relative_familiesPath)
    # print(relative_simList)
    print("relative_simNumber:{}".format(len(relative_familiesPath)))
    relative_SimPseudoInfo = []
    # relative_SimSplitVec = []
    # relative_SimImgVec = []
    # 找到与 blood_sim 的所有图片路径，以及 对应的伪标签路径
    for i in range(0, len(relative_familiesPath)):
        cur_imgPath = relative_familiesPath[i]
        cur_imgDatasetName = cur_imgPath.split("\\")[-3]
        cur_imgVideo = cur_imgPath.split("\\")[-2]
        cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]
        # 这里的label 和 Vec 都是图片内实例级别的
        cur_img_PseudoRank = os.path.join(BP_PseudoPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")
        # cur_img_splitVec = os.path.join(Bp_insVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")
        # cur_img_imgVec = os.path.join(BP_imgVecPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")

        relative_SimPseudoInfo.append(cur_img_PseudoRank)
        # relative_SimSplitVec.append(cur_img_splitVec)
        # relative_SimImgVec.append(cur_img_imgVec)

    inner_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))
    intra_knowledge_matrix = np.zeros((len(COCO_cat), len(COCO_cat)))
    # inner_knowledge_matrix = inner_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo,
    #                                               inner_knowledge_matrix, blood_SimSplitVec)
    # intra_knowledge_matrix = intra_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo,
    #                                               intra_knowledge_matrix, blood_SimSplitVec, blood_SimImgVec)

    inner_knowledge_matrix = inner_imageKnowledge(relative_familiesPath, relative_simList, relative_SimPseudoInfo,
                                                  inner_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)
    intra_knowledge_matrix = intra_imageKnowledge(relative_familiesPath, relative_simList, relative_SimPseudoInfo,
                                                  intra_knowledge_matrix, Bp_insVecPath, BP_imgVecPath)

    # 读取 最终80维 的矩阵
    general_knowledge_matrix = combine_InnerAndIntra(inner_knowledge_matrix, intra_knowledge_matrix)

    return inner_knowledge_matrix, intra_knowledge_matrix, general_knowledge_matrix


