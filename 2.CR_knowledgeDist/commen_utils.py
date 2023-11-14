import os
import numpy as np



COCO_cat = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# COCO_cat中的列表随意组合，得到COCO_Cat之间两两关系的组合


# # 关系的组合
# def Match(COCO_cat):
#     combinations = []
#     for i in range(len(COCO_cat)):
#         for j in range(i + 1, len(COCO_cat)):
#             combinations.append([COCO_cat[i], COCO_cat[j]])
#     return combinations
#
# COCO_catList = Match(COCO_cat)



# COCO_cat_sort = COCO_cat.copy()
# COCO_cat_sort = COCO_cat_sort.sort()

def read_TestDataset(PathTXT):

    file = open(PathTXT, "r", encoding="UTF-8")
    lines = file.readlines()
    file.close()
    All_imgPath = []

    for line_index in range(0, len(lines)):
        All_imgPath.append(lines[line_index].strip("\n"))

    return All_imgPath

def readPseudo_rank(path):

    file = open(path, "r", encoding="UTF-8")
    lines = file.readlines()
    file.close()

    img_All_objects = []
    for i in range(0, len(lines) - 1):
        img_All_objects.append(lines[i].split(",")[0])
    img_All_objects.sort()

    return img_All_objects

def find_top_n_indices(arr, n):
    indices = np.argsort(arr)[-n:][::-1]
    return indices

# 将相似矩阵中原本范围在(-1,1)之间的数，归一化到（0，1）之间
def normalize_matrix(matrix):
    # min_val = np.min(matrix)
    min_val = -1
    # print(min_val)
    # max_val = np.max(matrix)
    max_val = 1
    # print(max_val)

    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def cosine_similarity(vector, matrix):
    dot_product = np.dot(matrix, vector)
    norm_vector = np.linalg.norm(vector)
    norm_matrix = np.linalg.norm(matrix, axis=1)
    similarities = dot_product / (norm_vector * norm_matrix)
    return similarities


# 得到一个数组中所有元素的两两关系，保证每一个两两关系数组中两个元素的索引按输入数组的递增
def two_two_Relation(input_list):

    output_list = []

    for index in range(0, len(input_list)):

        current_object = input_list[index]
        for inner_index in range(index + 1, len(input_list)):
            current_inner_list = []
            current_inner_list.append(current_object)
            current_inner_list.append(input_list[inner_index])

            output_list.append(current_inner_list)

    return output_list

def normMatrix(input_matrix):
    norm = np.linalg.norm(input_matrix)
    normalized_matrix = input_matrix / norm
    return normalized_matrix

# 图内和图间的知识矩阵，怎么结合
set_innerWei = 0.5
set_intraWei = 0.5

def combine_InnerAndIntra(inner_knowledge_matrix, intra_knowledge_metrix):

    if inner_knowledge_matrix.shape == intra_knowledge_metrix.shape:
        # 首先对两个矩阵归一化，然后再相加
        inner_knowledge_matrix_norm = normMatrix(inner_knowledge_matrix)
        intra_knowledge_metrix_norm = normMatrix(intra_knowledge_metrix)

        final_group_matrix = set_innerWei * inner_knowledge_matrix_norm + set_intraWei * intra_knowledge_metrix_norm

    return final_group_matrix




# 传入矩阵以及对应的
def read_final_KnowledgeMatrix(cur_matrix, img_AllObjects):

    row_sum = np.sum(cur_matrix, axis=1)
    # 这里的列num指的是 想 x < y的次数
    column_sum = np.sum(cur_matrix, axis=0)
    total_num = row_sum + column_sum
    final_weights = row_sum / total_num
    final_weights = np.round(final_weights, decimals=2)
    for i in range(0, len(final_weights)):
        if isinstance(final_weights[i], (int, float)) == False:
            print("{}".format())
    # print(final_weights)
    finalDict = dict(zip(img_AllObjects, final_weights))
    finalOrder = sorted(finalDict.items(), key=lambda x: x[1], reverse=True)

    final_nameOrder = []
    final_score = []

    if len(finalOrder) == 1:
        cur_name, cur_weight = finalOrder[0]
        final_nameOrder.append(cur_name)
        final_score.append(float(1))
    else:
        for i in range(0, len(finalOrder)):
            cur_name, cur_weight = finalOrder[i]
            final_nameOrder.append(cur_name)
            final_score.append(cur_weight)

    return final_nameOrder, final_score


def readInsOrder(path):
    # 在这里进行排序
    labelInfo = open(path, "r", encoding="UTF-8")
    # 所有测试数据的信息
    lines = labelInfo.readlines()
    labelInfo.close()
    ins_nameList = []
    score_list = []

    for line_index in range(0, len(lines)):
        line_info = lines[line_index].split(",")
        ins_name = line_info[0]
        prior_score = float(line_info[1].split(":")[1])
        ins_nameList.append(ins_name)
        score_list.append(prior_score)

    ins_dict = dict(zip(ins_nameList, score_list))
    ins_dict = sorted(ins_dict.items(), key=lambda x: x[1], reverse=True)

    new_nameList, new_scoreList = [], []
    for i in range(0, len(ins_dict)):
        cur_name, cur_score = ins_dict[i]
        new_nameList.append(cur_name)
        new_scoreList.append(cur_score)

    return new_nameList, new_scoreList

def ins_Level_TTRelation(nameOrder, valueOrder):
    # 拆成具有先后顺序的两两排序列表
    new_nameList = []
    set_valueOrder = valueOrder.copy()
    set_valueOrder = sorted(list(set(set_valueOrder)), reverse=True)

    for index in range(0, len(set_valueOrder)):
        cur_value = set_valueOrder[index]
        inner_list = []
        for i in range(0, len(valueOrder)):
            if cur_value == valueOrder[i]:
                inner_list.append(nameOrder[i])

        new_nameList.append(inner_list)

    # 打成两两排序
    relation_list = []
    for i in range(0, len(new_nameList)):
        former_list = new_nameList[i]
        start = i + 1
        if i + 1 <= len(new_nameList):
            for index in range(start, len(new_nameList)):
                last_list = new_nameList[index]
                for former_index in range(0, len(former_list)):
                    former_ele = former_list[former_index]
                    for last_index in range(0, len(last_list)):
                        last_ele = last_list[last_index]
                        ele_list = []
                        ele_list.append(former_ele)
                        ele_list.append(last_ele)
                        relation_list.append(ele_list)
    # 这里得到的 relation_list 中的每一个小元素列表中有两个元素，且保证前一个元素显著性大于后一个元素
    return relation_list



def convertToWeiList(scoreList):
    sum_score = sum(scoreList)

    # 将列表转换为NumPy数组
    my_array = np.array(scoreList)
    result_array = my_array / sum_score
    result_list = result_array.tolist()

    return result_list


def cosine_similarity_vec(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)


def matrixToCatRank(Know_Matrix, img_allCats):

    # COCO_cat
    # img_allCats 是当前测试图片中所有的类别，只读取当前的矩阵中对应的类别知识就可以了
    test_CatIndexInCOCO_cat = []
    for cat_index in range(0, len(img_allCats)):
        cur_cat = img_allCats[cat_index]
        cur_CatIndex = COCO_cat.index(cur_cat)
        test_CatIndexInCOCO_cat.append(cur_CatIndex)


    # 将矩阵中对应的行列的关系读出
    # img_allCats 和 test_CatIndexInCOCO_cat对应
    cats_Wei = []
    for i in range(0, len(test_CatIndexInCOCO_cat)):
        # cur_cat = img_allCats[i]
        cur_index = test_CatIndexInCOCO_cat[i]

        row_sum = np.sum(Know_Matrix[cur_index])
        column_sum = np.sum(Know_Matrix[:, cur_index])

        cur_cat_Wei = row_sum / (row_sum + column_sum)
        cats_Wei.append(cur_cat_Wei)


    cat_orderDict = dict(zip(img_allCats, cats_Wei))
    cat_orderDict = sorted(cat_orderDict.items(), key=lambda x: x[1], reverse=True)

    final_catNames = []
    final_catScore = []
    for i in range(0, len(cat_orderDict)):
        pro_cat, pro_score = cat_orderDict[i]
        final_catNames.append(pro_cat)
        final_catScore.append(pro_score)



    return final_catNames, final_catScore



def saveFinalVotoRank_info(file_path, img_name, voteRank_Name, voteRank_score):

    path = os.path.join(file_path, img_name + ".txt")
    for index in range(0, len(voteRank_Name)):
        cur_name = voteRank_Name[index]
        cur_score = voteRank_score[index]
        cur_score = round(cur_score, 2)

        text_save = f"{cur_name},PriorScore:{cur_score}\n"
        f = open(path, 'a')
        f.write(text_save)
        f.close()

    frame_order = dict(zip(voteRank_Name, voteRank_score))
    frame_Dict = sorted(frame_order.items(), key=lambda x: x[1], reverse=True)
    frame_order = "frame_order!" + str({k: v for k, v in frame_Dict}) + "\n"

    f = open(path, 'a')
    f.write(frame_order)
    f.close()

def save_know_matrix(save_path, image_name, matrix):
    save_imgPath = os.path.join(save_path, image_name + ".npy")
    np.save(save_imgPath, matrix)