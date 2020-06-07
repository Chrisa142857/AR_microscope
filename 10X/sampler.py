from labelReader import labelReader as lread
from slideCroper import slideCroper as scrop
from config import cfg
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd
import sys
import os

W = 1936
H = 1216
img_save_path = 'Z:/wei/TCTDATA/detection/imgs_1936'
microType=sys.argv[1]  #3d our szsq
slideType=sys.argv[2]  #sfyX tjX
csv_fn = './detection_label_' + microType + '_' + slideType + '_' + str(W) +'.csv'
config3d1 = cfg(microType=microType, slideType=slideType)

def gen_neg_topleft(dataIds):
    img_fns = {}; imgs_num = 0
    topleft = {}; rand_coor = 10000; num_per_slide = 109
    scale = config3d1.scale

    for dataId in dataIds:
        if dataId not in topleft: topleft[dataId] = []
        if dataId not in img_fns: img_fns[dataId] = []
        slide = scrop(config3d1.slide_dir, dataId + config3d1.slide_tail, level=0)
        w, h = slide.get_wh()

        for _ in range(num_per_slide):
            sample_w = W*scale; sample_h = H*scale
            cropx = np.random.randint(int((w-rand_coor)/2), int((w+rand_coor)/2 - sample_w))
            cropy = np.random.randint(int((h-rand_coor)/2), int((h+rand_coor)/2 - sample_h))
            topleft[dataId].append([cropx, cropy, sample_w, sample_h])
            img_fn = slideType+'_'+str(dataId)+'_'+str(cropx)+'_'+str(cropy)+'_'+str(sample_w)+'_'+str(sample_h)+'.jpg'
            img_fns[dataId].append(img_fn)
            imgs_num += 1

    print('----------Generate ' + str(imgs_num) + ' NEG imgs----------')
    return topleft, img_fns

def gen_csv(dataIds):
    img_fns = {}; sample_xywh = {}; imgs_num = 0
    paths=[]; x_min=[]; y_min=[]; x_max=[]; y_max=[]; cls=[]
    scale = config3d1.scale
    pbar = tqdm(total=len(dataIds))
    lreader = lread()

    for dataId in dataIds:   
        labels = lreader.read(config3d1, dataId)
        if labels is None: 
            continue
        sample_xywh[dataId] = []; img_fns[dataId] = []
        pbar.update(1)
        for x, y, w, h, _ in labels:
            sample_w = int(W*scale); sample_h = int(H*scale)
            ddw = (sample_w - w); ddh = (sample_h - h)

            if ddw <= 0: ddw = 1
            if ddh <= 0: ddh = 1
            xrandmin = x - ddw; xrandmax = x
            yrandmin = y - ddh; yrandmax = y
            sample_x = np.random.randint(xrandmin, xrandmax)
            sample_y = np.random.randint(yrandmin, yrandmax)
            img_fn = slideType+'_'+str(dataId)+'_'+str(sample_x)+'_'+str(sample_y)+'_'+str(sample_w)+'_'+str(sample_h)+'.jpg'
            marksave_path = img_save_path + '/' + img_fn
            n_labels = 0

            for pxi, pyi, pwi, phi, cc in labels:

                if pxi>=sample_x and pyi>=sample_y and pxi+pwi<=sample_x+sample_w and pyi+phi<=sample_y+sample_h: #inner label
                    paths.append(marksave_path)
                    xxmin=int((pxi-sample_x)/scale); yymin=int((pyi-sample_y)/scale)
                    xxmax=int((pxi-sample_x+pwi)/scale); yymax=int((pyi-sample_y+phi)/scale)
                    x_min.append(xxmin); y_min.append(yymin)
                    x_max.append(xxmax); y_max.append(yymax)
                    cls.append(cc) 
                    n_labels += 1

            if n_labels != 0:
                sample_xywh[dataId].append([sample_x,sample_y,sample_w,sample_h])
                img_fns[dataId].append(img_fn)
                imgs_num += 1
    data = [[paths[i], x_min[i], y_min[i], x_max[i],y_max[i],cls[i]] for i in range(len(paths))]
    
    print('Output CSV with ' + str(imgs_num) + ' imgs and ' + str(len(data)) + ' labels in ' + csv_fn)
    col = ['path', 'xmin', 'ymin','xmax','ymax','cls']
    df = pd.DataFrame(data,columns=col)
    df.to_csv(csv_fn,index=False)        

    return sample_xywh, img_fns

def gen_imgs(name, config3d1, dataIds, sample_xywh, img_fns):

    # bar = tqdm(total = len(dataIds))
    # bar.set_description(name + ' Croping...')
    
    for dataId in dataIds:
        slide = scrop(config3d1.slide_dir, dataId + config3d1.slide_tail, level=0)
        # bar.update(1)

        for i in range(len(sample_xywh[dataId])):
            sample_x, sample_y, sample_w, sample_h = sample_xywh[dataId][i]
            img_fn = img_fns[dataId][i]
            tile_img = slide.crop(sample_x, sample_y, sample_w, sample_h)

            if tile_img is None: continue
            scaled_sample_img = cv2.resize(tile_img, (W, H))
            marksave_path = img_save_path + '/' + img_fn

            if not(cv2.imwrite(marksave_path, scaled_sample_img)):
                print(marksave_path + ' is existed ! Will be removed...')
                os.remove(marksave_path)
                cv2.imwrite(marksave_path, scaled_sample_img)


class myProcess(mp.Process):

    def __init__(self, threadID, name, config3d1, dataIds, sample_xywh, img_fns):
        mp.Process.__init__(self)
        self.threadID = threadID
        self.name = name
        self.dataIds = dataIds
        self.sample_xywh = sample_xywh
        self.img_fns = img_fns
        self.config3d1 = config3d1

    def run(self):
        print ("Start croping Process： " + self.name)
        gen_imgs(self.name, self.config3d1, self.dataIds, self.sample_xywh, self.img_fns)
        print ("End croping Process：" + self.name)

def main():
    dataIds = config3d1.dataIds
    
    if config3d1.label_dir != 'neg':
        sample_xywh, img_fns = gen_csv(dataIds)
    else:
        sample_xywh, img_fns = gen_neg_topleft(dataIds)
    cp = list(range(0, len(dataIds), 1+int(len(dataIds)/15)))
    cp[-1] = len(dataIds)
    threads = []

    for i in range(len(cp)-1):
        th = myProcess(i+1, "Process-"+str(i+1), config3d1, dataIds[cp[i]:cp[i+1]], sample_xywh, img_fns)
        threads.append(th)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()