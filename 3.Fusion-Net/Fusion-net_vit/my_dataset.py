
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import get_predInfo, COCO_cat, concat_npy, matrix_augment



matrix_Shuffle = 1
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, matrix_Shuffle=True):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.matrix_Shuffle = matrix_Shuffle

    def __len__(self):
        return len(self.images_path)


    def __getitem__(self, item):
        cur_npyPath = self.images_path[item]
        cur_labelPath = self.images_class[item]

        cur_imgName = cur_npyPath[0][0].split("\\")[-1].split(".")[0]


        blood_matrix, relative_matrix = concat_npy(cur_npyPath)

        blood_matrix = torch.tensor(blood_matrix)
        relative_matrix = torch.tensor(relative_matrix)
        blood_matrix = blood_matrix.float()
        relative_matrix = relative_matrix.float()


        GT_name, PriorScore = get_predInfo(cur_labelPath)

        final_label = [0] * 80

        for i in range(0, len(GT_name)):
            cur_cat = GT_name[i]
            indexInCOCO = COCO_cat.index(cur_cat)
            final_label[indexInCOCO] = len(GT_name) - i

        final_label = np.array(final_label)
        ori_label = final_label.copy()

        final_label = torch.tensor(final_label, dtype=torch.float32)

        if self.matrix_Shuffle != 0:
            shuffle_bloodList, shuffle_relativeList, shuffle_labelList = matrix_augment(cur_npyPath, ori_label, COCO_cat, self.matrix_Shuffle)
            shuffle_bloodList.append(blood_matrix)
            shuffle_relativeList.append(relative_matrix)
            shuffle_labelList.append(final_label)

            return shuffle_bloodList, shuffle_relativeList, shuffle_labelList, cur_imgName

        else:
            return blood_matrix, relative_matrix, final_label, cur_imgName

    @staticmethod
    def collate_fn(batch):
        if matrix_Shuffle != 0:
            blood_matrix_list, relative_matrix_list, labels, img_NameList = tuple(zip(*batch))
            all_blood, all_relative, all_labels = (), (), ()
            for index in range(0, len(blood_matrix_list)):
                cur_BloodList = blood_matrix_list[index]
                cur_RelativeList = relative_matrix_list[index]
                cur_LabelList = labels[index]

                for inner_index in range(0, len(cur_BloodList)):
                    all_blood = (*all_blood, cur_BloodList[inner_index])
                    all_relative = (*all_relative, cur_RelativeList[inner_index])
                    all_labels = (*all_labels, cur_LabelList[inner_index])



            blood_matrix = torch.stack(all_blood, dim=0)
            relative_matrix = torch.stack(all_relative, dim=0)
            labels = torch.stack(all_labels, dim=0)

            return blood_matrix, relative_matrix, labels, img_NameList

        else:

            blood_matrix_list, relative_matrix_list, labels, img_NameList = tuple(zip(*batch))
            blood_matrix = torch.stack(blood_matrix_list, dim=0)
            relative_matrix = torch.stack(relative_matrix_list, dim=0)
            labels = torch.stack(labels, dim=0)

            return blood_matrix, relative_matrix, labels, img_NameList

    @staticmethod
    def collate_fn_files(batch):

        blood_matrix_list, relative_matrix_list, labels, img_NameList = tuple(zip(*batch))
        all_blood, all_relative, all_labels = (), (), ()
        for index in range(0, len(blood_matrix_list)):
            cur_BloodList = blood_matrix_list[index]
            cur_RelativeList = relative_matrix_list[index]
            cur_LabelList = labels[index]

            for inner_index in range(0, len(cur_BloodList)):
                all_blood = (*all_blood, cur_BloodList[inner_index])
                all_relative = (*all_relative, cur_RelativeList[inner_index])
                all_labels = (*all_labels, cur_LabelList[inner_index])

        blood_matrix = torch.stack(all_blood, dim=0)
        relative_matrix = torch.stack(all_relative, dim=0)
        labels = torch.stack(all_labels, dim=0)

        return blood_matrix, relative_matrix, labels, img_NameList

    @staticmethod
    def collate_fn2(batch):
        blood_matrix_list, relative_matrix_list, labels, img_NameList = tuple(zip(*batch))
        blood_matrix = torch.stack(blood_matrix_list, dim=0)
        relative_matrix = torch.stack(relative_matrix_list, dim=0)
        labels = torch.stack(labels, dim=0)

        return blood_matrix, relative_matrix, labels, img_NameList