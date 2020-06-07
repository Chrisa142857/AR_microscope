from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/custom/images", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny-4.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints_4_nonsquare/yolov3_ckpt_1024_40.pth", help="path to weights file")
    parser.add_argument("--data_config", type=str, default="config/tj3&4_pos.data", help="path to data config file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=(1936,1216), help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, default="checkpoints_4_nonsquare/yolov3_ckpt_1024_2.pth", help="path to checkpoint model")
    parser.add_argument("--data_reg", default=False)
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    data_config = parse_data_config(opt.data_config)
    # train_path = data_config["train"]
    # dataset = ListDataset(train_path, augment=True, img_size=opt.img_size, multiscale=False)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.n_cpu,
    #     pin_memory=True,
    #     collate_fn=dataset.collate_fn,
    # )
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, reg=opt.data_reg, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    ii = 0
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        dataId = img_paths[0][img_paths[0].rfind('\\')+1:-4]
        if dataId.find('tj3') ==-1 and dataId.find('tj4') == -1: continue
        if dataId.find('neg') != -1: continue
        if ii == 5: break
        ii += 1
        input_imgs = Variable(input_imgs.type(Tensor))
        dt1 = datetime.datetime.now()
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        dt2 = datetime.datetime.now()
        print(dt2-dt1)
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        dataId = path[path.rfind('\\')+1:-4]
        gt_path = "data/custom/labels/" + dataId +".txt"
        with open(gt_path, 'r') as file:
            lines = file.readlines()
            gt = [line[:-1].split(' ')[1:] for line in lines]

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            # detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1-box_w/2, y1-box_h/2), box_w, box_h, linewidth=2, edgecolor=(1, 0, 0, 1), facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                # plt.text(
                #     x1,
                #     y1,
                #     s=classes[int(cls_pred)],
                #     color="white",
                #     verticalalignment="top",
                #     bbox={"color": color, "pad": 0},
                # )

        h, w = img.shape[:2]
        if len(gt[0]) != 0:
            gts = [[float(g[0]) * w, float(g[1]) * h, float(g[2]) * w, float(g[3]) * h] for g in gt]
            for gx, gy, gw, gh in gts:
                bbox = patches.Rectangle((gx - (gw / 2), gy - (gh / 2)), gw, gh, linewidth=1, edgecolor=(0, 1, 0, 1),
                                         facecolor="none")
                ax.add_patch(bbox)
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0, dpi=350)
        plt.close()
