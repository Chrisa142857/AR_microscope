import pandas as pd
import os
import xml.etree.ElementTree as ET

class labelReader:

    def __init__(self):
        self.label_name_list = ['LSIL_0.100', 'ASC-US', 'HSIL', '2211', '3000', 'HSIL_0.100', 'HISL', '2213', 'ASCUS',
         'SCC', 'ASC-US_0.100', 'HSIL_0.0', 'LSIL_0.0', 'LISL', 'LSIL', 'AGC', 'ASC-H', 'ASC-H_0.0',
         'ASC-US_0.0', '2230', 'ASC-H_0.100', 'pos', '2212', '2222', 'ASCH']
    
    def class_filter(self, cls_name):
        if cls_name in self.label_name_list:
            return 'pos'
        return None
    
    def read(self, cfg, dataId):

        tail = cfg.tail
        xywhcls_list = []
        if tail == '.csv':
            bbox_csv = pd.read_csv(cfg.label_dir + dataId + '/file2.csv', header=None)
            for idx in range(len(bbox_csv)):
                cls = self.class_filter(bbox_csv[0][idx])
                if cls is None:
                    continue
                tlx = bbox_csv[2][idx]
                tly = bbox_csv[3][idx]
                w = bbox_csv[4][idx]
                h = bbox_csv[5][idx]
                xywhcls_list.append([tlx, tly, w, h, cls])
        elif tail == '.xml':
            tree = ET.parse(cfg.label_dir + dataId + tail)
            root = tree.getroot()
            for ann in root.iter('Annotation'):
                cls = self.class_filter(ann.get('PartOfGroup'))
                if cls is None:
                    continue
                xs = [int(float(c.get('X'))) for c in ann.iter('Coordinate')]
                ys = [int(float(c.get('Y'))) for c in ann.iter('Coordinate')]
                if len(xs) < 2: continue
                tlx = min(xs)
                tly = min(ys)
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                xywhcls_list.append([tlx, tly, w, h, cls])

        return xywhcls_list