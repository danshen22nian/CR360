import os
import cv2
import torch
import numpy as np
from PIL import Image

def cv_read_img(img):
    src = cv2.imread(img)
    if src is None:
        # QMessageBox.warning(self, "载入出错", "图片读取失败。\n（可能原因：无图片、无正确权限、不受支持或未知的格式）")
        return None
    return src

def count_nonzero_points(arr):
    count = np.count_nonzero(arr)
    return count

def sum_array_elements(arr):
    total_sum = np.sum(arr)
    return total_sum

def find_nonzero_indices(arr):
    nonzero_indices = np.nonzero(arr)
    return np.transpose(nonzero_indices)

def find_common_nonzero_indices(arr1, arr2):
    common_indices = np.transpose(np.nonzero((arr1 != 0) & (arr2 != 0)))
    return common_indices
ERP_objSegPath = r"E:\final_project\rank_project\vector_process\img_insERPSeg"
dataset_names = os.listdir(ERP_objSegPath)
dataset_names = ["360_SOD", "AOI", "F_360iSOD", "No360"]
salmap_path = r"D:\A_CatRank\trans_cube\erp_sal"
rank_filePath = r"D:\A_CatRank\trans_cube\rank_file"

for data_index in range(0, len(dataset_names)):
    seg_dataPath = os.path.join(ERP_objSegPath, dataset_names[data_index])
    video_names = os.listdir(seg_dataPath)
    salmap_dataPath = os.path.join(salmap_path, dataset_names[data_index])

    save_rankPath = os.path.join(rank_filePath, dataset_names[data_index])

    for video_index in range(0 ,len(video_names)):
        print("{} / {}".format(video_index, len(video_names)))
        Seg_video_path = os.path.join(seg_dataPath, video_names[video_index])
        Seg_frames = os.listdir(Seg_video_path)

        sal_video_path = os.path.join(salmap_dataPath, video_names[video_index])
        salFrames = os.listdir(sal_video_path)

        save_VideoTXTPath = os.path.join(save_rankPath, video_names[video_index])
        if not os.path.exists(save_VideoTXTPath):  # 判在文件夹如果不存在则创建为文件夹
            os.makedirs(save_VideoTXTPath)


        for frame_index in range(0, len(Seg_frames)):

            seg_framePath = os.path.join(Seg_video_path, Seg_frames[frame_index])
            object_names = os.listdir(seg_framePath)
            sal_frame = os.path.join(sal_video_path, Seg_frames[frame_index] + ".png")

            image = Image.open(sal_frame)
            width, height = image.size
            save_frameTXT = os.path.join(save_VideoTXTPath, Seg_frames[frame_index] + ".txt")
            All_objectNames = []
            PriorScore = []

            for object_index in range(0, len(object_names)):
                object_segPath = os.path.join(seg_framePath, object_names[object_index])
                object_name = object_names[object_index].split(".")[0]


                sal_read = cv2.imread(sal_frame, cv2.IMREAD_GRAYSCALE)
                class_segImg = np.array(cv2.resize(cv2.imread(object_segPath, cv2.IMREAD_GRAYSCALE), (width, height)),
                                        dtype=np.float32)
                whole_zone = count_nonzero_points(class_segImg)

                common_indexes = find_common_nonzero_indices(class_segImg, sal_read)

                img_np = np.zeros_like(sal_read)
                sum = 0
                for index in range(0, len(common_indexes)):
                    x, y = common_indexes[index]

                    img_np[x][y] = sal_read[x][y]
                    sum = sum + sal_read[x][y]


                score = float(sum / whole_zone)
                score = round(score, 2)

                new_name = str(object_name.split(".")[0].split("_")[0]) + "_" + str(object_name.split(".")[0].split("_")[1])
                All_objectNames.append(new_name)
                PriorScore.append(score)

                score = round(score, 2)
                Prior_Score_str = "{:.2f}".format(score)

                text_save = f"{new_name},PriorScore:{Prior_Score_str}\n"

                f = open(save_frameTXT, 'a')
                f.write(text_save)
                f.close()








