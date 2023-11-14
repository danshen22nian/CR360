import os

import numpy as np

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]










# 正确的 encode
# 将其中的 N / A 都去掉

print(len(CLASSES))

del_ele = 'N/A'
NEW_classes = []
index_val = []
for index in range(0, len(CLASSES)):
    if CLASSES[index] != del_ele:
        index_val.append(index)
        NEW_classes.append(CLASSES[index])


print(len(NEW_classes))
print(len(index_val))

print(NEW_classes)

new_str = ""

for index in range(0, len(NEW_classes)):
    if index == len(NEW_classes) - 1:
        new_str = new_str + NEW_classes[index]
    else:
        new_str = new_str + NEW_classes[index] + ","

print(new_str)

class_str = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light," \
            "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear," \
            "zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite," \
            "baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon," \
            "bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table," \
            "toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors," \
            "teddy bear,hair drier,toothbrush"

# print("{}".format())




def gen_newNpy(ori_objects_npy, index_val):

    # print(ori_objects_npy)

    # 去掉无效的类别

    for index in range(0, len(ori_objects_npy)):
        cur_object_npy = ori_objects_npy[index]
        cur_new_list = []
        for inner_index in range(0, len(index_val)):
            cur_new_list.append(cur_object_npy[index_val[inner_index]])

        cur_new_npy = np.array(cur_new_list)

        if index == 0:
            final_npy = cur_new_npy
        else:
            final_npy = final_npy + cur_new_npy

        #  看看这一条现在表示的是什么 信息
    # print("***************")


    # print(final_npy.shape)
    # ind = np.argpartition(final_npy, -10)[-10:]

    # print(ind)
    # # ind = np.argpartition(ori_objects_npy, -5)[-5:]
    # cluster_class = []
    # value = []
    # for i in range(0, len(ind)):
    #     cluster_class.append(NEW_classes[ind[i]])
    #     value.append(final_npy[ind[i]])
    # print(cluster_class)
    # print(value)
    # print("{}".format())
    return final_npy






    # print("{}".format())



#
ori_object_npy = r"E:\360_dataset\F-360iSOD\my_Data\save_ERP_total_object_npy"
ori_object_npy = r"E:\final_project\360-test-557\other_files\dection\save_ERP_total_object_npy"
ori_object_npy = r"E:\final_project\dataset_needed\ASSR_toCatRank\dection_results\test\vector"
ori_object_npy = r"E:\final_project\IRSR_toCatRank_new\dection_results\test\vector"
ori_object_npy = r"E:\360_dataset\360-SOD\360-SOD\save_ERP_total_object_npy"
# ori_object_npy = r"E:\CommonTrain_UseFiles\new_npy\save_ERP_total_object_npy"
# ori_object_npy = r"E:\CommonTest_UseFiles\new_npy\save_ERP_total_object_npy"
# ori_object_npy = r"E:\360_dataset\AOI_Use\new_npy\save_ERP_total_object_npy"
ori_videoFiles = os.listdir(ori_object_npy)
#
# ori_tar = r"E:\360_dataset\F-360iSOD\my_Data\ERP_final_npy"
# ori_tar = r"E:\CommonTrain_UseFiles\new_npy\ERP_final_npy"
# ori_tar = r"E:\CommonTest_UseFiles\new_npy\ERP_final_npy"
#ori_tar = r"E:\360_dataset\AOI_Use\new_npy\ERP_final_npy"

new_tar = r"E:\final_project\360-test-557\other_files\dection\new_nnnnnewERP_final_npy"
new_tar = r"E:\final_project\IRSR_toCatRank_new\dection_results\test\final_npy"
new_tar = r"E:\360_dataset\360-SOD\360-SOD\final_npy"
# new_tar = r"E:\CommonTrain_UseFiles\new_npy\new_nnnnnewERP_final_npy"
# new_tar = r"E:\CommonTest_UseFiles\new_npy\new_nnnnnewERP_final_npy"
# new_tar = r"E:\360_dataset\AOI_Use\new_npy\new_nnnnnewERP_final_npy"


for video_index in range(0, len(ori_videoFiles)):
    video_path = os.path.join(ori_object_npy, ori_videoFiles[video_index])
    frame_files = os.listdir(video_path)
    save_videoPath = os.path.join(new_tar, ori_videoFiles[video_index])

    if not os.path.exists(save_videoPath):  # 判在文件夹如果不存在则创建为文件夹
        os.makedirs(save_videoPath)

    for frame_index in range(0, len(frame_files)):
        frame_path = os.path.join(video_path, frame_files[frame_index])
        # print(frame_path)
        # print("{}".format())
        save_frame_npy = os.path.join(save_videoPath, frame_files[frame_index])

        # print(save_frame_npy)
        # print("{]".format())

        # 原来对应的所有目标的
        ori_objects_npy = np.load(frame_path)

        # # 查看原来的final
        # ori_final_Path = os.path.join(ori_tar, ori_videoFiles[video_index], frame_files[frame_index])
        # ori_final = np.load(ori_final_Path)
        # ind = np.argpartition(ori_final, -10)[-10:]
        # # print(ind)
        # # ind = np.argpartition(ori_objects_npy, -5)[-5:]
        # cluster_class = []
        # for i in range(0, len(ind)):
        #     cluster_class.append(CLASSES[ind[i]])
        # print(ori_final_Path)
        # print(cluster_class)
        # # 共16 个目标
        # print(len(ori_objects_npy))



        final_new_npy = gen_newNpy(ori_objects_npy, index_val)
        np.save(save_frame_npy, final_new_npy)