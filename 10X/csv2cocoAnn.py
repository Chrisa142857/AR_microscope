import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
import sys

np.random.seed(51)
W = 1936
H = 1216
# 0为背景
classname_to_id = {'pos':1}

class Csv2CoCo:

    def __init__(self, image_dir, total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi, label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}

        image['height'] = W
        image['width'] = H
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):

        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x, min_y, min_x, min_y + 0.5 * h, min_x, max_y, min_x + 0.5 * w, max_y, max_x, max_y, max_x,
                  max_y - 0.5 * h, max_x, min_y, max_x - 0.5 * w, min_y])
        return a

if __name__ == '__main__':

    csv_file = "./detection_label_"+sys.argv[1]+"_"+sys.argv[2]+"_1936.csv"
    image_dir = "Z:/wei/TCTDATA/detection/imgs_1936"
    saved_coco_path = "Z:/wei/TCTDATA/detection/cocoAnns/"
    
    # 整合csv格式标注文件
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file, header=None).values
    
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value

    # 按照键值划分数
    total_keys = list(total_csv_annotations.keys())[1:]
    print(len(total_keys))
    # train_keys, val_keys = train_test_split(total_keys, test_size=0.05)
    # print("train_n:", len(train_keys), 'val_n:', len(val_keys))
    # 创建必须的
    ##########################################################################
    if not os.path.exists('%scoco/annotations/' % saved_coco_path):
        os.makedirs('%scoco/annotations/' % saved_coco_path)
    # 把训练集转化为COCO的json格式
    l2c_train = Csv2CoCo(image_dir=image_dir, total_annos=total_csv_annotations)
    train_instance = l2c_train.to_coco(total_keys)
    l2c_train.save_coco_json(train_instance, saved_coco_path + 'instances_' + sys.argv[1] + "_" + sys.argv[2] + '_2cls_1936.json')

    # # 把验证集转化为COCO的json格式
    # l2c_val = Csv2CoCo(image_dir=image_dir, total_annos=total_csv_annotations)
    # val_instance = l2c_val.to_coco(val_keys)
    # l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017_16cls.json' % saved_coco_path)
    