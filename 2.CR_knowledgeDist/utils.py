import os
import numpy as np
from commen_utils import *
from distill_utils import *

def InitAllData(BP_imgPath, BP_PseudoPath, BP_imgVecPath, Bp_insVecPath, BP_mainFile):
    print("Base Pool loading----------------------------------")
    basePoolImgPath = []
    data_files = os.listdir(BP_imgPath)

    All_infoNum = 0
    val_infoNum = 0

    for file_index in range(0, len(data_files)):
        filePath = os.path.join(BP_imgPath, data_files[file_index])
        videoFiles = os.listdir(filePath)
        print(filePath)
        cur_dataset_imgPath = []
        cur_allInfoNum = 0
        cur_valInfoNum = 0
        save_cur_imgPathTXT = os.path.join(BP_mainFile, data_files[file_index]+ "_val" + ".txt")

        for video_index in range(0, len(videoFiles)):
            video_path = os.path.join(filePath, videoFiles[video_index])
            frame_names = os.listdir(video_path)
            for frame_index in range(0, len(frame_names)):
                All_infoNum = All_infoNum + 1
                cur_allInfoNum = cur_allInfoNum + 1
                frame_path = os.path.join(video_path, frame_names[frame_index])

                frame_labelPath = os.path.join(BP_PseudoPath, data_files[file_index], videoFiles[video_index], frame_names[frame_index].split(".")[0] + ".txt")
                frame_ImgVecPath = os.path.join(BP_imgVecPath, data_files[file_index], videoFiles[video_index], frame_names[frame_index].split(".")[0] + ".npy")
                frame_InsVecPath = os.path.join(Bp_insVecPath, data_files[file_index], videoFiles[video_index], frame_names[frame_index].split(".")[0] + ".npy")

                if os.path.exists(frame_labelPath) == True and \
                   os.path.exists(frame_ImgVecPath) == True and \
                   os.path.exists(frame_InsVecPath) == True:

                    val_infoNum = val_infoNum + 1
                    cur_valInfoNum = cur_valInfoNum + 1
                    basePoolImgPath.append(frame_path)
                    cur_dataset_imgPath.append(frame_path)

        print("{} have loaded， total_num {}， {} are valid".format(data_files[file_index], cur_allInfoNum, cur_valInfoNum))

        for i in range(0, len(cur_dataset_imgPath)):
            cur_imgPath = cur_dataset_imgPath[i]

            text_save = f"{cur_imgPath}\n"
            f = open(save_cur_imgPathTXT, 'a')
            f.write(text_save)
            f.close()


    basePoolImgPath.sort()
    save_imgPathTXT = os.path.join(BP_mainFile, "BP_imgInfo.txt")
    for i in range(0, len(basePoolImgPath)):
        cur_imgPath = basePoolImgPath[i]

        text_save = f"{cur_imgPath}\n"
        f = open(save_imgPathTXT, 'a')
        f.write(text_save)
        f.close()


    print("Base Pool have loaded，total_num {}, {} are valid".format(All_infoNum, val_infoNum))
    return basePoolImgPath



def get_baseDataNumpy(bpAll_imgPath, ImgVec_mainFile, BP_mainFile, BP_imgPath):
    dataset_Names = os.listdir(BP_imgPath)

    for img_index in range(0, len(bpAll_imgPath)):
        img_path = bpAll_imgPath[img_index]
        cur_imgDatasetName = img_path.split("\\")[-3]
        cur_imgVideo = img_path.split("\\")[-2]
        cur_imgName = img_path.split("\\")[-1].split(".")[0] + ".npy"
        if cur_imgDatasetName in dataset_Names:

            cur_imgVectorPath = os.path.join(ImgVec_mainFile, cur_imgDatasetName, cur_imgVideo, cur_imgName)

            if img_index == 0:
                cur_vec = np.load(cur_imgVectorPath)
                BPVecMatrix = cur_vec
            else:
                cur_vec = np.load(cur_imgVectorPath)
                BPVecMatrix = np.vstack((BPVecMatrix, cur_vec))

    save_bp_npy = os.path.join(BP_mainFile, "bp_vector.npy")
    np.save(save_bp_npy, BPVecMatrix)


    bp_npy = np.load(save_bp_npy)
    print(bp_npy.shape)

    print("Base Pool matrix saved in：{}".format(save_bp_npy))
    return BPVecMatrix


def Init_TestData(New_DataMainPath, test_Dataset):
    print("test data loading-------------------------")
    print("cur dataset： {}".format(test_Dataset))

    all_testNum = 0
    all_testValNUm = 0


    for data_index in range(0, len(test_Dataset)):
        data_imgPath = os.path.join(New_DataMainPath, "image", test_Dataset[data_index])
        data_labelPath = os.path.join(New_DataMainPath, "pseudo_label", test_Dataset[data_index])
        data_vecPath = os.path.join(New_DataMainPath, "imgVec", test_Dataset[data_index])

        data_VideoFiles = os.listdir(data_imgPath)
        cur_ValDataPath = []

        cur_infoNum = 0
        cur_valNum = 0

        for video_index in range(0, len(data_VideoFiles)):
            video_path = os.path.join(data_imgPath, data_VideoFiles[video_index])
            frame_names = os.listdir(video_path)

            for frame_index in range(0, len(frame_names)):
                cur_infoNum = cur_infoNum + 1
                frame_imgPath = os.path.join(video_path, frame_names[frame_index])
                frame_labelPath = os.path.join(data_labelPath, data_VideoFiles[video_index], frame_names[frame_index].split(".")[0] + ".txt")
                frame_vecPath = os.path.join(data_vecPath, data_VideoFiles[video_index], frame_names[frame_index].split(".")[0] + ".npy")

                if os.path.exists(frame_vecPath) == True and os.path.exists(frame_labelPath) == True:
                    cur_ValDataPath.append(frame_imgPath)
                    cur_valNum = cur_valNum + 1


        cur_ValDataPath.sort()
        save_imgPathTXT = os.path.join(New_DataMainPath, test_Dataset[data_index] + "_valData" + ".txt")

        for i in range(0, len(cur_ValDataPath)):
            cur_imgPath = cur_ValDataPath[i]

            text_save = f"{cur_imgPath}\n"
            f = open(save_imgPathTXT, 'a')
            f.write(text_save)
            f.close()

        print("{} have loaded，total num {},  {} are valid".format(test_Dataset[data_index], cur_infoNum, cur_valNum))
        print("-------------------------------------------------")

        all_testNum = all_testNum + cur_infoNum
        all_testValNUm = all_testValNUm + cur_valNum

    print("*********************")
    print("current test datasize: {}， {} are valid".format(all_testNum, all_testValNUm))


def bp_AllImgAndNpy(bp_imgPathTXT, bp_allDataNpy):

    file = open(bp_imgPathTXT, "r", encoding="UTF-8")
    lines = file.readlines()
    file.close()
    bp_All_ImgPath = []

    for line_index in range(0, len(lines)):
        bp_All_ImgPath.append(lines[line_index].strip("\n"))


    bp_npy = np.load(bp_allDataNpy)
    print("Base Pool data and semantic matrix have loaded，total_num {}，shape of matrix :{} ".format(len(bp_All_ImgPath), bp_npy.shape))
    return bp_All_ImgPath, bp_npy

def get_testImgCatRankOrder(cur_Dataset_AllImgs, test_mainFile, test_Datasetname, save_DataSet_results,
                            bp_All_ImgPath, bp_npy, set_firstSearch_per, set_secondSearch_per, BP_PseudoPath,BP_imgVecPath,Bp_insVecPath,
                            use_firstFilter, use_secondFilter):
    print("start testing----------------------------------------------")
    test_imgPath = os.path.join(test_mainFile, "image")
    test_PseudoPath = os.path.join(test_mainFile, "pseudo_label")
    test_vectorPath = os.path.join(test_mainFile, "imgVec")

    for img_index in range(0, len(cur_Dataset_AllImgs)):
        print("{} \ {}".format(img_index, len(cur_Dataset_AllImgs)))
        cur_imgPath = cur_Dataset_AllImgs[img_index]


        cur_imgDatasetName = cur_imgPath.split("\\")[-3]
        cur_imgVideo = cur_imgPath.split("\\")[-2]
        cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]


        cur_imgRankPath = os.path.join(test_PseudoPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")

        cur_imgVecPath = os.path.join(test_vectorPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")


        img_AllObjects = readPseudo_rank(cur_imgRankPath)
        if len(img_AllObjects) == 1:
            continue
        img_AllObjects.sort()


        cur_vec = np.load(cur_imgVecPath)

        sim_numpy = cosine_similarity(cur_vec, bp_npy)

        sim_numpy = normalize_matrix(sim_numpy)

        if set_firstSearch_per != None:
            most_sim_num = int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01))
            most_sim_index = find_top_n_indices(sim_numpy, most_sim_num)
        else:

            most_sim_index = find_top_n_indices(sim_numpy, len(sim_numpy))

        sim_list = list(most_sim_index)

        blood_SimImgPath = []
        blood_SimImgSimList = []


        for i in range(0, len(sim_list)):
            blood_SimImgPath.append(bp_All_ImgPath[sim_list[i]])
            blood_SimImgSimList.append(sim_numpy[sim_list[i]])


        if use_firstFilter == False and use_secondFilter == False:
            print("--")

        elif use_firstFilter == True and use_secondFilter == False:
            print("-")
        elif use_firstFilter == True and use_secondFilter == True:
            save_test_resultsPath = os.path.join(save_DataSet_results, "whole_testResults")
            blood_inner_matrix, blood_intra_matrix, expert_knowledge_matrix = blood_related_KnowlodgeDistill(
                blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath, Bp_insVecPath, BP_imgVecPath)
            print("blood_SimNumber:{}".format(len(blood_SimImgPath)))
            print("blood done!")
            used_Family_imgPath = []
            used_Family_imgPath = used_Family_imgPath + blood_SimImgPath
            relative_inner_matrix, relative_intra_matrix, general_knowledge_matrix = relative_related_knowledgeDistill(blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath,
                                                                     Bp_insVecPath, BP_imgVecPath, used_Family_imgPath, set_secondSearch_per, bp_npy, bp_All_ImgPath)
            print("relative done!")
            save_bloodInner_path = os.path.join(save_test_resultsPath, "blood", "inner_npy")
            save_bloodIntra_path = os.path.join(save_test_resultsPath, "blood", "intra_npy")
            save_relativeInner_path = os.path.join(save_test_resultsPath, "relative", "inner_npy")
            save_relativeIntra_path = os.path.join(save_test_resultsPath, "relative", "intra_npy")

            if not os.path.exists(save_bloodInner_path):
                os.makedirs(save_bloodInner_path)
            if not os.path.exists(save_bloodIntra_path):
                os.makedirs(save_bloodIntra_path)
            if not os.path.exists(save_relativeInner_path):
                os.makedirs(save_relativeInner_path)
            if not os.path.exists(save_relativeIntra_path):
                os.makedirs(save_relativeIntra_path)

            save_know_matrix(save_bloodInner_path, cur_imgName, blood_inner_matrix)
            save_know_matrix(save_bloodIntra_path, cur_imgName, blood_intra_matrix)
            save_know_matrix(save_relativeInner_path, cur_imgName, relative_inner_matrix)
            save_know_matrix(save_relativeIntra_path, cur_imgName, relative_intra_matrix)

            blood_inner_OrderName, blood_inner_OrderScore = matrixToCatRank(blood_inner_matrix, img_AllObjects)
            blood_intra_OrderName, blood_intra_OrderScore = matrixToCatRank(blood_intra_matrix, img_AllObjects)
            rel_inner_OrderName, rel_inner_OrderScore = matrixToCatRank(relative_inner_matrix, img_AllObjects)
            rel_intra_OrderName, rel_intra_OrderScore = matrixToCatRank(relative_intra_matrix, img_AllObjects)


            save_bloodInner_RankPath = os.path.join(save_test_resultsPath, "blood", "inner_rankFile")
            save_bloodIntra_RankPath  = os.path.join(save_test_resultsPath, "blood", "intra_rankFile")
            save_relativeInner_RankPath  = os.path.join(save_test_resultsPath, "relative", "inner_rankFile")
            save_relativeIntra_RankPath  = os.path.join(save_test_resultsPath, "relative", "intra_rankFile")

            if not os.path.exists(save_bloodInner_RankPath):
                os.makedirs(save_bloodInner_RankPath)
            if not os.path.exists(save_bloodIntra_RankPath):
                os.makedirs(save_bloodIntra_RankPath)
            if not os.path.exists(save_relativeInner_RankPath):
                os.makedirs(save_relativeInner_RankPath)
            if not os.path.exists(save_relativeIntra_RankPath):
                os.makedirs(save_relativeIntra_RankPath)

            saveFinalVotoRank_info(save_bloodInner_RankPath, cur_imgName, blood_inner_OrderName, blood_inner_OrderScore)
            saveFinalVotoRank_info(save_bloodIntra_RankPath, cur_imgName, blood_intra_OrderName, blood_intra_OrderScore)
            saveFinalVotoRank_info(save_relativeInner_RankPath, cur_imgName, rel_inner_OrderName, rel_inner_OrderScore)
            saveFinalVotoRank_info(save_relativeIntra_RankPath, cur_imgName, rel_intra_OrderName, rel_intra_OrderScore)