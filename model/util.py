# -*- coding: utf-8 -*-

import numpy as np
import math


def get_embeddings():
    file = ["fasttext.txt", "word2vec.txt", "all.txt"]
    embeddings = {}
    for i in file:
        dic = {}
        with open("model/" + i, "r") as f:
            for line in f.read().splitlines():
                name, vector = line.split(" ")
                # if name == "background":
                #    continue
                vector = vector.split(",")
                vector = list(map(float, vector))
                tmp = 0
                for x in vector:
                    tmp += x ** 2
                tmp = math.sqrt(tmp)
                dic[name] = [x / tmp for x in vector]
            embeddings[i.split(".")[0]] = dic
    return embeddings


def get_Ws(embeddings, strong_classes):
    file = ["fasttext", "word2vec", "all"]
    strong_len = len(strong_classes)
    Ws = {}
    for name in file:
        lenth = 300
        if name == "all":
            lenth = 600
        embedding = embeddings[name]
        strong = np.zeros([strong_len, lenth], dtype=np.float)
        weak = np.zeros([20 - strong_len, lenth], dtype=np.float)
        all = np.zeros([20, lenth], dtype=np.float)
        i, j, k = 0, 0, 0
        for class_name in embedding:
            if class_name == "background":
                continue
            if class_name in strong_classes:
                strong[i] = embedding[class_name]
                i += 1
            else:
                weak[j] = embedding[class_name]
                j += 1
            all[k] = embedding[class_name]
            k += 1
        Ws[name + "_strong"] = strong
        Ws[name + "_weak"] = weak
        Ws[name + "_all"] = all
    return Ws


def get_embeddings_coco():
    file = ["fasttext_coco.txt", "word2vec_coco.txt", "all_coco.txt"]
    embeddings = {}
    for i in file:
        dic = {}
        with open("model/" + i, "r") as f:
            for line in f.read().splitlines():
                name, vector = line.split(" ")
                vector = vector.split(",")
                vector = list(map(float, vector))
                # tmp = 0
                # for x in vector:
                #     tmp += x ** 2
                # tmp = math.sqrt(tmp)
                # dic[name] = [x / tmp for x in vector]
                dic[name] = vector
            embeddings[i.split("_")[0]] = dic
    return embeddings


def get_Ws_coco(embeddings):
    CLASSES_DIC = {"unlabeled": 255, "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4, "bus": 5,
                   "train": 6, "truck": 7, "boat": 8, "stoplight": 9, "fireplug": 10, "street_sign": 11,
                   "stop_sign": 12, "parking_meter": 13, "bench": 14, "bird": 15, "cat": 16, "dog": 17, "horse": 18,
                   "sheep": 19, "cow": 20, "elephant": 21, "bear": 22, "zebra": 23, "giraffe": 24, "hat": 25,
                   "backpack": 26, "umbrella": 27, "shoe": 28, "eye_glasses": 29, "handbag": 30, "tie": 31,
                   "suitcase": 32, "frisbee": 33, "skis": 34, "snowboard": 35, "sports_ball": 36, "kite": 37,
                   "baseball_bat": 38, "baseball_glove": 39, "skateboard": 40, "surfboard": 41, "tennis_racket": 42,
                   "bottle": 43, "plate": 44, "wineglass": 45, "cup": 46, "fork": 47, "knife": 48, "spoon": 49,
                   "bowl": 50, "banana": 51, "apple": 52, "sandwich": 53, "orange": 54, "broccoli": 55, "carrot": 56,
                   "hotdog": 57, "pizza": 58, "donut": 59, "cake": 60, "chair": 61, "couch": 62, "houseplant": 63,
                   "bed": 64, "mirror": 65, "dining_table": 66, "window": 67, "desk": 68, "toilet": 69, "door": 70,
                   "tv": 71, "laptop": 72, "mouse": 73, "remote": 74, "keyboard": 75, "mobilephone": 76,
                   "microwave": 77, "oven": 78, "toaster": 79, "sink": 80, "refrigerator": 81, "blender": 82,
                   "book": 83, "clock": 84, "vase": 85, "scissors": 86, "teddybear": 87, "blowdryer": 88,
                   "toothbrush": 89, "hair_brush": 90, "banner": 91, "blanket": 92, "branch": 93, "bridge": 94,
                   "building_other": 95, "bush": 96, "cabinet": 97, "cage": 98, "cardboard": 99, "carpet": 100,
                   "ceiling_other": 101, "ceiling_tile": 102, "cloth": 103, "clothes": 104, "clouds": 105,
                   "counter": 106, "cupboard": 107, "curtain": 108, "desk-stuff": 109, "dirt": 110, "door-stuff": 111,
                   "fence": 112, "floor_marble": 113, "floor_other": 114, "floor_stone": 115, "floor_tile": 116,
                   "floor_wood": 117, "flower": 118, "fog": 119, "food_other": 120, "fruit": 121,
                   "furniture_other": 122, "grass": 123, "gravel": 124, "ground_other": 125, "hill": 126, "house": 127,
                   "leaves": 128, "light": 129, "mat": 130, "metal": 131, "mirror-stuff": 132, "moss": 133,
                   "mountain": 134, "mud": 135, "napkin": 136, "net": 137, "paper": 138, "pavement": 139, "pillow": 140,
                   "plant_other": 141, "plastic": 142, "platform": 143, "playground": 144, "railing": 145,
                   "railroad": 146, "river": 147, "road": 148, "rock": 149, "roof": 150, "rug": 151, "salad": 152,
                   "sand": 153, "sea": 154, "shelf": 155, "sky_other": 156, "skyscraper": 157, "snow": 158,
                   "solid_other": 159, "stairs": 160, "stone": 161, "straw": 162, "structural_other": 163, "table": 164,
                   "tent": 165, "textile_other": 166, "towel": 167, "tree": 168, "vegetable": 169, "wall_brick": 170,
                   "wall_concrete": 171, "wall_other": 172, "wall_panel": 173, "wall_stone": 174, "wall_tile": 175,
                   "wall_wood": 176, "water_other": 177, "water_drops": 178, "window_blind": 179, "window_other": 180,
                   "wood": 181}
    RELABELED_CLASSES_DIC = {255: 255, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 12: 11,
                             13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 21: 19, 22: 20, 23: 21, 26: 22,
                             27: 23, 30: 24, 31: 25, 34: 26, 35: 27, 36: 28, 37: 29, 38: 30, 39: 31, 41: 32, 42: 33,
                             43: 34, 45: 35, 46: 36, 47: 37, 48: 38, 49: 39, 50: 40, 51: 41, 52: 42, 53: 43, 54: 44,
                             55: 45, 57: 46, 58: 47, 59: 48, 60: 49, 61: 50, 62: 51, 63: 52, 64: 53, 66: 54, 69: 55,
                             71: 56, 72: 57, 73: 58, 74: 59, 75: 60, 76: 61, 77: 62, 78: 63, 79: 64, 80: 65, 81: 66,
                             83: 67, 84: 68, 85: 69, 87: 70, 88: 71, 89: 72, 91: 73, 92: 74, 93: 75, 94: 76, 95: 77,
                             96: 78, 97: 79, 98: 80, 100: 81, 101: 82, 102: 83, 103: 84, 104: 85, 106: 86, 107: 87,
                             108: 88, 109: 89, 110: 90, 111: 91, 112: 92, 113: 93, 114: 94, 115: 95, 116: 96,
                             117: 97, 118: 98, 119: 99, 120: 100, 121: 101, 122: 102, 124: 103, 125: 104, 126: 105,
                             127: 106, 128: 107, 129: 108, 130: 109, 131: 110, 132: 111, 133: 112, 134: 113,
                             135: 114, 136: 115, 137: 116, 138: 117, 139: 118, 140: 119, 141: 120, 142: 121,
                             143: 122, 145: 123, 146: 124, 149: 125, 150: 126, 151: 127, 152: 128, 153: 129,
                             154: 130, 155: 131, 156: 132, 157: 133, 158: 134, 159: 135, 160: 136, 161: 137,
                             162: 138, 163: 139, 164: 140, 165: 141, 166: 142, 167: 143, 169: 144, 170: 145,
                             172: 146, 173: 147, 174: 148, 175: 149, 176: 150, 177: 151, 178: 152, 179: 153,
                             180: 154, 181: 155, 33: 156, 40: 157, 99: 158, 56: 159, 86: 160, 32: 161, 24: 162,
                             148: 163, 171: 164, 20: 165, 168: 166, 123: 167, 147: 168, 105: 169, 144: 170, 11: 255,
                             25: 255, 28: 255, 29: 255, 44: 255, 65: 255, 67: 255, 68: 255, 70: 255, 82: 255,
                             90: 255}
    REMOVED_CLASSES_DIC = {"unlabeled": 255, "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
                           "bus": 5,
                           "train": 6, "truck": 7, "boat": 8, "stoplight": 9, "fireplug": 10, "stop_sign": 12,
                           "parking_meter": 13, "bench": 14, "bird": 15, "cat": 16, "dog": 17, "horse": 18, "sheep": 19,
                           "cow": 20, "elephant": 21, "bear": 22, "zebra": 23, "giraffe": 24, "backpack": 26,
                           "umbrella": 27, "handbag": 30, "tie": 31, "suitcase": 32, "frisbee": 33, "skis": 34,
                           "snowboard": 35, "sports_ball": 36, "kite": 37, "baseball_bat": 38, "baseball_glove": 39,
                           "skateboard": 40, "surfboard": 41, "tennis_racket": 42, "bottle": 43, "wineglass": 45,
                           "cup": 46,
                           "fork": 47, "knife": 48, "spoon": 49, "bowl": 50, "banana": 51, "apple": 52, "sandwich": 53,
                           "orange": 54, "broccoli": 55, "carrot": 56, "hotdog": 57, "pizza": 58, "donut": 59,
                           "cake": 60,
                           "chair": 61, "couch": 62, "houseplant": 63, "bed": 64, "dining_table": 66, "toilet": 69,
                           "tv": 71, "laptop": 72, "mouse": 73, "remote": 74, "keyboard": 75, "mobilephone": 76,
                           "microwave": 77, "oven": 78, "toaster": 79, "sink": 80, "refrigerator": 81, "book": 83,
                           "clock": 84, "vase": 85, "scissors": 86, "teddybear": 87, "blowdryer": 88, "toothbrush": 89,
                           "banner": 91, "blanket": 92, "branch": 93, "bridge": 94, "building_other": 95, "bush": 96,
                           "cabinet": 97, "cage": 98, "cardboard": 99, "carpet": 100, "ceiling_other": 101,
                           "ceiling_tile": 102, "cloth": 103, "clothes": 104, "clouds": 105, "counter": 106,
                           "cupboard": 107, "curtain": 108, "desk-stuff": 109, "dirt": 110, "door-stuff": 111,
                           "fence": 112,
                           "floor_marble": 113, "floor_other": 114, "floor_stone": 115, "floor_tile": 116,
                           "floor_wood": 117, "flower": 118, "fog": 119, "food_other": 120, "fruit": 121,
                           "furniture_other": 122, "grass": 123, "gravel": 124, "ground_other": 125, "hill": 126,
                           "house": 127, "leaves": 128, "light": 129, "mat": 130, "metal": 131, "mirror-stuff": 132,
                           "moss": 133, "mountain": 134, "mud": 135, "napkin": 136, "net": 137, "paper": 138,
                           "pavement": 139, "pillow": 140, "plant_other": 141, "plastic": 142, "platform": 143,
                           "playground": 144, "railing": 145, "railroad": 146, "river": 147, "road": 148, "rock": 149,
                           "roof": 150, "rug": 151, "salad": 152, "sand": 153, "sea": 154, "shelf": 155,
                           "sky_other": 156,
                           "skyscraper": 157, "snow": 158, "solid_other": 159, "stairs": 160, "stone": 161,
                           "straw": 162,
                           "structural_other": 163, "table": 164, "tent": 165, "textile_other": 166, "towel": 167,
                           "tree": 168, "vegetable": 169, "wall_brick": 170, "wall_concrete": 171, "wall_other": 172,
                           "wall_panel": 173, "wall_stone": 174, "wall_tile": 175, "wall_wood": 176, "water_other": 177,
                           "water_drops": 178, "window_blind": 179, "window_other": 180, "wood": 181}

    weak_classes = ["frisbee", "skateboard", "cardboard", "carrot", "scissors", "suitcase", "giraffe", "road",
                    "wall_concrete", "cow", "tree", "grass", "river", "clouds", "playground"]

    strong_classes = [x for x in REMOVED_CLASSES_DIC if x not in weak_classes]
    strong_classes.remove("unlabeled")

    file = ["all", "fasttext", "word2vec"]
    strong_len = len(strong_classes)
    Ws = {}
    for name in file:
        lenth = 300
        if name == "all":
            lenth = 600
        embedding = embeddings[name]
        strong = np.zeros([strong_len, lenth], dtype=np.float)
        weak = np.zeros([171 - strong_len, lenth], dtype=np.float)
        all = np.zeros([171, lenth], dtype=np.float)
        for class_name in embedding:
            if RELABELED_CLASSES_DIC[CLASSES_DIC[class_name]] == 255:
                continue
            if class_name in strong_classes:
                strong[RELABELED_CLASSES_DIC[CLASSES_DIC[class_name]]] = embedding[class_name]
            elif class_name in weak_classes:
                weak[RELABELED_CLASSES_DIC[CLASSES_DIC[class_name]] - 156] = embedding[class_name]
            all[RELABELED_CLASSES_DIC[CLASSES_DIC[class_name]]] = embedding[class_name]
        Ws[name + "_strong"] = strong
        Ws[name + "_weak"] = weak
        Ws[name + "_all"] = all
    return Ws


def get_Ws_split(embeddings, split):
    ALL_CLASSES = ["bg", "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                   "table", "dog", "horse", "motorbike", "person", "houseplant", "sheep", "sofa", "train",
                   "monitor"]
    all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    split1 = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    split2 = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    split3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
    split4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    strong_class = []
    if split == "1":
        strong_class = [ALL_CLASSES[x] for x in split1]
    if split == "2":
        strong_class = [ALL_CLASSES[x] for x in split2]
    if split == "3":
        strong_class = [ALL_CLASSES[x] for x in split3]
    if split == "4":
        strong_class = [ALL_CLASSES[x] for x in split4]

    file = ["all", "fasttext", "word2vec"]
    strong_len = 15
    Ws = {}
    for name in file:
        lenth = 300
        if name == "all":
            lenth = 600
        embedding = embeddings[name]
        strong = np.zeros([strong_len, lenth], dtype=np.float)
        weak = np.zeros([20 - strong_len, lenth], dtype=np.float)

        i, j = 0, 0
        for class_name in embedding:
            if class_name == "background":
                continue
            if class_name in strong_class:
                strong[i] = embedding[class_name]
                i += 1
            else:
                weak[j] = embedding[class_name]
                j += 1
        all = np.concatenate([strong,weak])

        Ws[name + "_strong"] = strong
        Ws[name + "_weak"] = weak
        Ws[name + "_all"] = all
    return Ws
