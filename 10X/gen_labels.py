import json
import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

data_path = 'data/custom/'
save_path = 'data/custom/labels/'
img_path = 'data/custom/images/'
w = 1936
h = 1216

def json2txts(s, train_str, valid_str):
    images = s['images']
    annotations = s['annotations']
    i = 0
    img_flag = {}
    for ann in annotations:
        img_fn = images[ann['image_id']]['file_name']
        img_flag[img_fn] = 0

    for ann in annotations:
        img_fn = images[ann['image_id']]['file_name']

        box = list(map(lambda e, f: e * f, ann['bbox'], [1. / w, 1. / h, 1. / w, 1. / h]))

        # img = np.array(Image.open(img_fn))
        # plt.figure()
        # fig, ax = plt.subplots(1)
        # ax.imshow(img)
        # g = box
        # # h, w = img.shape[:2]
        # gts = [[float(g[0]) * w, float(g[1]) * h, float(g[2]) * w, float(g[3]) * h]]
        # for gx, gy, gw, gh in gts:
        #     bbox = patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor=(0, 1, 0, 1),
        #                              facecolor="none")
        #     ax.add_patch(bbox)
        # # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = img_fn.split("/")[-1].split(".")[0]
        # plt.savefig(f"exp_fig/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        # plt.close()

        f = img_fn[img_fn.rfind('/') + 1:-4] + '.txt'
        if os.path.exists(img_path + f[:-4] + '.jpg'):
            if img_flag[img_fn] == 0:
                img_flag[img_fn] = 1
                if i >= 10:
                    i = 0
                    valid_str += img_path + f[:-4] + '.jpg\n'
                else:
                    i += 1
                    train_str += img_path + f[:-4] + '.jpg\n'

            with open(save_path + f, "a") as file:
                file.write('0 ' + str(box[0]+(box[2]/2)) + ' ' + str(box[1]+(box[3]/2)) + ' ' + str(box[2]) + ' ' + str(box[3]) + '\n')

    return train_str, valid_str

def gen_data_txt(str, txt_name):

    with open(data_path + txt_name, "w") as file:
        file.write(str)

def neg_json2txts(flag, train_str, valid_str):
    i = 0
    img_fns = []
    for r, d, fs in os.walk(img_path):
        break
    for f in fs:
        if f.find(flag) != -1:
            img_fns.append(f)

    for img_fn in img_fns:

        f = img_fn[img_fn.rfind('/') + 1:-4] + '.txt'
        if os.path.exists(img_path + f[:-4] + '.jpg'):

            if i >= 10:
                i = 0
                valid_str += img_path + f[:-4] + '.jpg\n'
            else:
                i += 1
                train_str += img_path + f[:-4] + '.jpg\n'

            with open(save_path + f, "w") as file:
                file.write('\n')

    return train_str, valid_str

train_str = ''
valid_str = ''
train_str, valid_str = neg_json2txts('tj3_neg', train_str, valid_str)
train_str, valid_str = neg_json2txts('tj4_neg', train_str, valid_str)

train_txt_name = 'tj3&4_neg_train.txt'
valid_txt_name = 'tj3&4_neg_valid.txt'

gen_data_txt(train_str, train_txt_name)
gen_data_txt(valid_str, valid_txt_name)

train_str = ''
valid_str = ''
train_str, valid_str = neg_json2txts('sfy1_neg', train_str, valid_str)
train_str, valid_str = neg_json2txts('sfy2_neg', train_str, valid_str)

train_txt_name = 'sfy1&2_neg_train.txt'
valid_txt_name = 'sfy1&2_neg_valid.txt'

gen_data_txt(train_str, train_txt_name)
gen_data_txt(valid_str, valid_txt_name)

s1s = json.load(open('Z:/wei/TCTDATA/detection/cocoAnns/instances_szsq_tj3_pos_2cls_1936.json', 'r'))
s2s = json.load(open('Z:/wei/TCTDATA/detection/cocoAnns/instances_szsq_tj4_pos_2cls_1936.json', 'r'))

train_str = ''
valid_str = ''
train_str, valid_str = json2txts(s1s, train_str, valid_str)
train_str, valid_str = json2txts(s2s, train_str, valid_str)

train_txt_name = 'tj3&4_pos_train.txt'
valid_txt_name = 'tj3&4_pos_valid.txt'

gen_data_txt(train_str, train_txt_name)
gen_data_txt(valid_str, valid_txt_name)