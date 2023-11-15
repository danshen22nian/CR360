
from commen_utils import *

intra_relThreshold = 0.2
intra_relSimThreshold = 0.7


def intra_imageKnowledge(blood_SimImgPath, blood_SimImgSimList, blood_SimPseudoInfo, intra_knowledge_matrix, Bp_insVecPath, BP_imgVecPath):

    for out_index in range(0, len(blood_SimImgPath)):
        print("{} \ {}".format(out_index, len(blood_SimImgPath)))
        cur_anchorImgPath = blood_SimImgPath[out_index]
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


        anc_ins_weight = convertToWeiList(anc_ins_score)

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


            cur_comImg_lebelPath = blood_SimPseudoInfo[inner_index]
            com_ins_name, com_ins_score = readInsOrder(cur_comImg_lebelPath)

            com_flag = False
            for i in range(0, len(com_ins_score)):
                if com_ins_score[i] != 0:
                    com_flag = True
            if com_flag == False or len(com_ins_name) <= 1:
                continue


            com_ins_weight = convertToWeiList(com_ins_score)

            cur_comImgVecPath = os.path.join(BP_imgVecPath, cur_comImgDatasetName, cur_comImgVideo, cur_comImgName + ".npy")
            cur_comInsVecPath = os.path.join(Bp_insVecPath, cur_comImgDatasetName, cur_comImgVideo, cur_comImgName + ".npy")
            cur_comImgVec = np.load(cur_comImgVecPath)
            cur_comInsVec = np.load(cur_comInsVecPath)



            ancAndcom_sim = cosine_similarity_vec(cur_comImgVec, cur_ancImgVec)

            if intra_relSimThreshold != None:
                if ancAndcom_sim < intra_relSimThreshold:
                    continue

            ancAndcom_matchList = [[a, b] for a in anc_ins_name for b in com_ins_name]

            for rel_index in range(0, len(ancAndcom_matchList)):
                rel_ancInsName = ancAndcom_matchList[rel_index][0]
                rel_comInsName = ancAndcom_matchList[rel_index][1]

                anc_insIndexInWeight = anc_ins_name.index(rel_ancInsName)
                ancInsWeight = anc_ins_weight[anc_insIndexInWeight]
                com_insIndexInWeight = com_ins_name.index(rel_comInsName)
                comInsWeight = com_ins_weight[com_insIndexInWeight]


                ancInsIndexInVec = int(rel_ancInsName.split("_")[1])
                comInsIndexInVec = int(rel_comInsName.split("_")[1])

                ancInsVec = cur_ancInsVec[ancInsIndexInVec]
                comInsVec = cur_comInsVec[comInsIndexInVec]


                cur_rel_Weight = abs(ancInsWeight - comInsWeight)
                if cur_rel_Weight >= intra_relThreshold:

                    if ancInsWeight > comInsWeight:

                        big_insVec = ancInsVec
                        small_insVec = comInsVec

                    else:
                        big_insVec = comInsVec
                        small_insVec = ancInsVec

                    big_insVec = big_insVec.reshape((80, 1))
                    small_insVec = small_insVec.reshape((1, 80))
                    cur_img_KnowMatrix = np.dot(big_insVec, small_insVec)

                    cur_img_KnowMatrix = cur_img_KnowMatrix * ancAndcom_sim * cur_rel_Weight

                    intra_knowledge_matrix = intra_knowledge_matrix + cur_img_KnowMatrix
                    np.fill_diagonal(intra_knowledge_matrix, 0)


    return intra_knowledge_matrix
