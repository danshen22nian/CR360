
def two_twoRelation(nameOrder, valueOrder):

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

    return relation_list

