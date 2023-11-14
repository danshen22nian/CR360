import os
import numpy as np
from commen_utils import *
from distill_utils import *

# 初始化
def InitAllData(BP_imgPath, BP_PseudoPath, BP_imgVecPath, Bp_insVecPath, BP_mainFile):
    print("Base Pool 数据加载----------------------------------")
    basePoolImgPath = []
    data_files = os.listdir(BP_imgPath)

    # print("{}".format())
    All_infoNum = 0
    val_infoNum = 0

    for file_index in range(0, len(data_files)):
        filePath = os.path.join(BP_imgPath, data_files[file_index])
        videoFiles = os.listdir(filePath)
        print(filePath)

        # 将每一个数据集的有效数据也保存一下
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
                # only save those have objects
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

        print("{} 数据集加载完毕， 共 {} 条数据，其中共 {} 条数据中是有目标出现的".format(data_files[file_index], cur_allInfoNum, cur_valInfoNum))

        for i in range(0, len(cur_dataset_imgPath)):
            cur_imgPath = cur_dataset_imgPath[i]
            # write every path into txt file
            text_save = f"{cur_imgPath}\n"
            f = open(save_cur_imgPathTXT, 'a')
            f.write(text_save)
            f.close()

    # sort and save all
    basePoolImgPath.sort()
    save_imgPathTXT = os.path.join(BP_mainFile, "BP_imgInfo.txt")
    for i in range(0, len(basePoolImgPath)):
        cur_imgPath = basePoolImgPath[i]
        # write every path into txt file
        text_save = f"{cur_imgPath}\n"
        f = open(save_imgPathTXT, 'a')
        f.write(text_save)
        f.close()


    print("Base Pool 数据加载完毕，共 {} 条数据, 其中共 {} 条数据中是有目标出现的".format(All_infoNum, val_infoNum))
    return basePoolImgPath



def get_baseDataNumpy(bpAll_imgPath, ImgVec_mainFile, BP_mainFile, BP_imgPath):
    dataset_Names = os.listdir(BP_imgPath)
    # print(dataset_Names)
    # 遍历所有的路径
    for img_index in range(0, len(bpAll_imgPath)):
        img_path = bpAll_imgPath[img_index]
        # print(img_path)
        # print("{}".format())
        # 判断属于哪个数据集的
        cur_imgDatasetName = img_path.split("\\")[-3]
        cur_imgVideo = img_path.split("\\")[-2]
        cur_imgName = img_path.split("\\")[-1].split(".")[0] + ".npy"
        if cur_imgDatasetName in dataset_Names:
            # 找到它的语义向量
            cur_imgVectorPath = os.path.join(ImgVec_mainFile, cur_imgDatasetName, cur_imgVideo, cur_imgName)
            # print(cur_imgVectorPath)
            # print("{}".format())
            # 读取语义向量
            if img_index == 0:
                # 以第一条数据的语义向量初始化
                cur_vec = np.load(cur_imgVectorPath)
                BPVecMatrix = cur_vec
            else:
                cur_vec = np.load(cur_imgVectorPath)
                BPVecMatrix = np.vstack((BPVecMatrix, cur_vec))

    save_bp_npy = os.path.join(BP_mainFile, "bp_vector.npy")
    np.save(save_bp_npy, BPVecMatrix)

    # test
    bp_npy = np.load(save_bp_npy)
    print(bp_npy.shape)

    print("Base Pool 向量矩阵已保存在：{}".format(save_bp_npy))
    return BPVecMatrix


def Init_TestData(New_DataMainPath, test_Dataset):
    print("加载测试数据-------------------------")
    print("此次要测试的数据集： {}".format(test_Dataset))

    all_testNum = 0
    all_testValNUm = 0

    # 将最后的有效路径保存在
    for data_index in range(0, len(test_Dataset)):
        data_imgPath = os.path.join(New_DataMainPath, "image", test_Dataset[data_index])
        data_labelPath = os.path.join(New_DataMainPath, "pseudo_label", test_Dataset[data_index])
        data_vecPath = os.path.join(New_DataMainPath, "imgVec", test_Dataset[data_index])

        data_VideoFiles = os.listdir(data_imgPath)
        cur_ValDataPath = []

        # 当前数据集共多少数据，多少有效数据
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

        # 写数据
        cur_ValDataPath.sort()
        save_imgPathTXT = os.path.join(New_DataMainPath, test_Dataset[data_index] + "_valData" + ".txt")

        for i in range(0, len(cur_ValDataPath)):
            cur_imgPath = cur_ValDataPath[i]
            # write every path into txt file
            text_save = f"{cur_imgPath}\n"
            f = open(save_imgPathTXT, 'a')
            f.write(text_save)
            f.close()

        print("{} 数据集加载完毕，共 {} 条数据, 其中共 {} 条数据中是由目标出现的".format(test_Dataset[data_index], cur_infoNum, cur_valNum))
        print("-------------------------------------------------")

        all_testNum = all_testNum + cur_infoNum
        all_testValNUm = all_testValNUm + cur_valNum

    print("*********************")
    print("此次测试共 {} 条数据， 其中共 {} 有效".format(all_testNum, all_testValNUm))


# 加载BP池数
# 读取 base pool 中的所有数据，以及它的语义向量的npy
def bp_AllImgAndNpy(bp_imgPathTXT, bp_allDataNpy):

    file = open(bp_imgPathTXT, "r", encoding="UTF-8")
    lines = file.readlines()
    file.close()
    bp_All_ImgPath = []

    for line_index in range(0, len(lines)):
        bp_All_ImgPath.append(lines[line_index].strip("\n"))

    # 读取bp语义向量
    bp_npy = np.load(bp_allDataNpy)
    print("Base Pool 数据及语义矩阵加载完毕，共 {} 条数据，矩阵 shape :{} ".format(len(bp_All_ImgPath), bp_npy.shape))
    return bp_All_ImgPath, bp_npy

def get_testImgCatRankOrder(cur_Dataset_AllImgs, test_mainFile, test_Datasetname, save_DataSet_results,
                            bp_All_ImgPath, bp_npy, set_firstSearch_per, set_secondSearch_per, BP_PseudoPath,BP_imgVecPath,Bp_insVecPath,
                            use_firstFilter, use_secondFilter):
    print("开始预测测试数据----------------------------------------------")
    # 直接给定了类别
    test_imgPath = os.path.join(test_mainFile, "image")
    test_PseudoPath = os.path.join(test_mainFile, "pseudo_label")
    test_vectorPath = os.path.join(test_mainFile, "imgVec")

    for img_index in range(0, len(cur_Dataset_AllImgs)):
        print("{} \ {}".format(img_index, len(cur_Dataset_AllImgs)))
        cur_imgPath = cur_Dataset_AllImgs[img_index]

        # 图片信息
        cur_imgDatasetName = cur_imgPath.split("\\")[-3]
        cur_imgVideo = cur_imgPath.split("\\")[-2]
        cur_imgName = cur_imgPath.split("\\")[-1].split(".")[0]

        # 类别和图片语义向量
        cur_imgRankPath = os.path.join(test_PseudoPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")
        # 图片级别的向量用于 索引相似场景
        cur_imgVecPath = os.path.join(test_vectorPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".npy")

        # 当前测试图片中的所有类别
        img_AllObjects = readPseudo_rank(cur_imgRankPath)
        if len(img_AllObjects) == 1:
            continue
        img_AllObjects.sort()

        # 测试图片语义向量
        cur_vec = np.load(cur_imgVecPath)
        # 然后求相似性得到一个表示矩阵
        sim_numpy = cosine_similarity(cur_vec, bp_npy)
        # 对一个矩阵中所有（-1，1) 的数归一化到（0，1）之间
        sim_numpy = normalize_matrix(sim_numpy)

        if set_firstSearch_per != None:
            most_sim_num = int(np.floor(len(bp_All_ImgPath) * set_firstSearch_per * 0.01))
            most_sim_index = find_top_n_indices(sim_numpy, most_sim_num)
        else:
            # 参考所有的
            most_sim_index = find_top_n_indices(sim_numpy, len(sim_numpy))

        sim_list = list(most_sim_index)
        # 保存所有的路径，以及 路径所对应的sim值作为，第一个置信度
        blood_SimImgPath = []
        blood_SimImgSimList = []

        # 得到最终相似度由大到小的对比顺序
        for i in range(0, len(sim_list)):
            blood_SimImgPath.append(bp_All_ImgPath[sim_list[i]])
            blood_SimImgSimList.append(sim_numpy[sim_list[i]])


        if use_firstFilter == False and use_secondFilter == False:
            print("--")
            # save_test_resultsPath = os.path.join(save_DataSet_results, "noDistill_testResults")
            # save_test_resultsComparePath = os.path.join(save_DataSet_results, "noDistil_testCompare.txt")
            # final_OrderName, final_OrderScore = noKnowlegdeDistill(blood_SimImgPath, blood_SimImgSimList, img_AllObjects, BP_mainFile)

        elif use_firstFilter == True and use_secondFilter == False:
            print("-")
            # save_test_resultsPath = os.path.join(save_DataSet_results, "FirstFilter_testResults")
            # # save_test_resultsComparePath = os.path.join(save_DataSet_results, "FirstFilter_testCompare.txt")
            #
            # inner_knowledge_matrix, intra_knowledge_matrix, expert_knowledge_matrix = blood_related_KnowlodgeDistill(blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath, Bp_insVecPath, BP_imgVecPath)
            # # matrixToCatRank
            #
            # inner_OrderName, inner_OrderScore = matrixToCatRank(inner_knowledge_matrix, img_AllObjects)
            # intra_OrderName, intra_OrderScore = matrixToCatRank(intra_knowledge_matrix, img_AllObjects)
            #
            # # 读取矩阵中的知识
            #
            # final_OrderName, final_OrderScore = matrixToCatRank(expert_knowledge_matrix, img_AllObjects)
            #
            # save_innerResultsPath = os.path.join(save_test_resultsPath, "inner")
            # save_intraResultsPath = os.path.join(save_test_resultsPath, "intra")
            # save_combinePath = os.path.join(save_test_resultsPath, "combine")
            # if not os.path.exists(
            #         os.path.join(save_innerResultsPath, cur_imgDatasetName, cur_imgVideo)):  # 判在文件夹如果不存在则创建为文件夹
            #     os.makedirs(os.path.join(save_innerResultsPath, cur_imgDatasetName, cur_imgVideo))
            # if not os.path.exists(
            #         os.path.join(save_intraResultsPath, cur_imgDatasetName, cur_imgVideo)):  # 判在文件夹如果不存在则创建为文件夹
            #     os.makedirs(os.path.join(save_intraResultsPath, cur_imgDatasetName, cur_imgVideo))
            # if not os.path.exists(
            #         os.path.join(save_combinePath, cur_imgDatasetName, cur_imgVideo)):  # 判在文件夹如果不存在则创建为文件夹
            #     os.makedirs(os.path.join(save_combinePath, cur_imgDatasetName, cur_imgVideo))
            #
            # save_innerResults_imgPath = os.path.join(save_innerResultsPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")
            # save_intraResults_imgPath = os.path.join(save_intraResultsPath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")
            # save_CombineResults_imgPath = os.path.join(save_combinePath, cur_imgDatasetName, cur_imgVideo, cur_imgName + ".txt")
            #
            # saveFinalVotoRank_info(save_innerResults_imgPath, inner_OrderName, inner_OrderScore)
            # saveFinalVotoRank_info(save_intraResults_imgPath, intra_OrderName, intra_OrderScore)
            # saveFinalVotoRank_info(save_CombineResults_imgPath, final_OrderName, final_OrderScore)



        elif use_firstFilter == True and use_secondFilter == True:

            save_test_resultsPath = os.path.join(save_DataSet_results, "whole_testResults")
            # save_test_bloodPath = os.path.join(save_DataSet_results, "blood_process", cur_imgName)
            # save_test_relativePath = os.path.join(save_DataSet_results, "relative_process", cur_imgName)


            blood_inner_matrix, blood_intra_matrix, expert_knowledge_matrix = blood_related_KnowlodgeDistill(
                blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath, Bp_insVecPath, BP_imgVecPath)
            print("blood_SimNumber:{}".format(len(blood_SimImgPath)))
            print("blood done!")
            # 用过的关系
            used_Family_imgPath = []
            used_Family_imgPath = used_Family_imgPath + blood_SimImgPath
            relative_inner_matrix, relative_intra_matrix, general_knowledge_matrix = relative_related_knowledgeDistill(blood_SimImgPath, blood_SimImgSimList, BP_PseudoPath,
                                                                     Bp_insVecPath, BP_imgVecPath, used_Family_imgPath, set_secondSearch_per, bp_npy, bp_All_ImgPath)
            print("relative done!")

            # whole_testResults/blood/inner or intra
            # whole_testResults/relative
            save_bloodInner_path = os.path.join(save_test_resultsPath, "blood", "inner_npy")
            save_bloodIntra_path = os.path.join(save_test_resultsPath, "blood", "intra_npy")
            save_relativeInner_path = os.path.join(save_test_resultsPath, "relative", "inner_npy")
            save_relativeIntra_path = os.path.join(save_test_resultsPath, "relative", "intra_npy")

            if not os.path.exists(save_bloodInner_path):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_bloodInner_path)
            if not os.path.exists(save_bloodIntra_path):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_bloodIntra_path)
            if not os.path.exists(save_relativeInner_path):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_relativeInner_path)
            if not os.path.exists(save_relativeIntra_path):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_relativeIntra_path)

            save_know_matrix(save_bloodInner_path, cur_imgName, blood_inner_matrix)
            save_know_matrix(save_bloodIntra_path, cur_imgName, blood_intra_matrix)
            save_know_matrix(save_relativeInner_path, cur_imgName, relative_inner_matrix)
            save_know_matrix(save_relativeIntra_path, cur_imgName, relative_intra_matrix)

            blood_inner_OrderName, blood_inner_OrderScore = matrixToCatRank(blood_inner_matrix, img_AllObjects)
            blood_intra_OrderName, blood_intra_OrderScore = matrixToCatRank(blood_intra_matrix, img_AllObjects)
            rel_inner_OrderName, rel_inner_OrderScore = matrixToCatRank(relative_inner_matrix, img_AllObjects)
            rel_intra_OrderName, rel_intra_OrderScore = matrixToCatRank(relative_intra_matrix, img_AllObjects)


            # save_reuslts
            save_bloodInner_RankPath = os.path.join(save_test_resultsPath, "blood", "inner_rankFile")
            save_bloodIntra_RankPath  = os.path.join(save_test_resultsPath, "blood", "intra_rankFile")
            save_relativeInner_RankPath  = os.path.join(save_test_resultsPath, "relative", "inner_rankFile")
            save_relativeIntra_RankPath  = os.path.join(save_test_resultsPath, "relative", "intra_rankFile")

            if not os.path.exists(save_bloodInner_RankPath):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_bloodInner_RankPath)
            if not os.path.exists(save_bloodIntra_RankPath):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_bloodIntra_RankPath)
            if not os.path.exists(save_relativeInner_RankPath):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_relativeInner_RankPath)
            if not os.path.exists(save_relativeIntra_RankPath):  # 判在文件夹如果不存在则创建为文件夹
                os.makedirs(save_relativeIntra_RankPath)

            saveFinalVotoRank_info(save_bloodInner_RankPath, cur_imgName, blood_inner_OrderName, blood_inner_OrderScore)
            saveFinalVotoRank_info(save_bloodIntra_RankPath, cur_imgName, blood_intra_OrderName, blood_intra_OrderScore)
            saveFinalVotoRank_info(save_relativeInner_RankPath, cur_imgName, rel_inner_OrderName, rel_inner_OrderScore)
            saveFinalVotoRank_info(save_relativeIntra_RankPath, cur_imgName, rel_intra_OrderName, rel_intra_OrderScore)