import os
import os.path as osp


def get_model_path(path):
    model_list = os.listdir(path)
    num_list = [int(x.split("_")[0]) for x in model_list if "model" in x]
    max_num = str(max(list(map(int, num_list))))
    name_set = set()
    name_list = [x.split("_")[1] for x in model_list if "_" in x]
    name_list = [x.split(".")[0] for x in name_list]
    for i in name_list:
        name_set.add(i)
    name_dict = {"step": int(max_num)}
    for name in name_set:
        model = osp.join(path, max_num + "_" + name + ".pth")
        name_dict[name] = model

    return name_dict


def delete_superfluous_model(path, max_num):
    model_list = os.listdir(path)
    num_list = [int(x.split("_")[0]) for x in model_list if "model" in x]
    name_set = set()
    name_list = [x.split("_")[1] for x in model_list if "_" in x]
    name_list = [x.split(".")[0] for x in name_list]
    for i in name_list:
        name_set.add(i)

    while len(num_list) > max_num:
        min_num = str(min(num_list))
        for i in name_set:
            model = osp.join(path, min_num + "_" + i + ".pth")
            os.remove(model)
            print("delete %s\n" % (model))
        num_list.remove(int(min_num))
