from models import *
from utils.utils import *
from utils.datasets import *
import argparse
import os
import datetime
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def get_tp_recall(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives,[detected_box.numpy() for detected_box in detected_boxes]])
    return batch_metrics


def f(model, img_size, batch_size, list_path, conf_thres=0.1, nms_thres=0.5, iou_thres=0.5):
    UNSURE_LABEL_NUM = 0
    POS_LABEL_NUM = 0
    with open(list_path, "r") as file:
        img_files = file.readlines()

    label_files = [
        path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        for path in img_files
    ]
    unsureLabel_files = [
        path.replace("images", "labels").replace(".png", "_withUnsure.txt").replace(".jpg", "_withUnsure.txt")
        for path in img_files
    ]

    model.eval()

    # Get dataloader
    dataset = ListDataset(list_path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # labels = []
    # sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        label_path = label_files[batch_i % len(img_files)].rstrip()
        unsureLabel_path = unsureLabel_files[batch_i % len(img_files)].rstrip()
        # labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # Concatenate sample statistics
        res = get_tp_recall(outputs, targets, iou_threshold=iou_thres)

        saving_str = ''
        with open(unsureLabel_path, 'w') as saving:
            if len(res) == 0:
                unsure_boxes = targets[:, 2:]
                for b in unsure_boxes:
                    b = [coor.item()/img_size for coor in b]
                    UNSURE_LABEL_NUM += 1
                    saving_str += "1 " + str((b[0]+b[2])/2) + ' ' + str((b[1]+b[3])/2) + ' ' + str(b[2]-b[0]) + ' ' + str(b[3]-b[1]) + '\n'
            else:
                for id, (TP, detected_boxes) in enumerate(res, 0):
                    ind = np.zeros(targets.shape[0])
                    if len(detected_boxes) != 0:
                        ind[np.stack(detected_boxes)] = 1
                    pos_boxes = targets[ind==1, 2:]
                    unsure_boxes = targets[ind==0, 2:]
                    for b in unsure_boxes:
                        b = [coor.item()/img_size for coor in b]
                        UNSURE_LABEL_NUM += 1
                        saving_str += "1 " + str((b[0]+b[2])/2) + ' ' + str((b[1]+b[3])/2) + ' ' + str(b[2]-b[0]) + ' ' + str(b[3]-b[1]) + '\n'
                    for b in pos_boxes:
                        b = [coor.item()/img_size for coor in b]
                        POS_LABEL_NUM += 1
                        saving_str += "0 " + str((b[0]+b[2])/2) + ' ' + str((b[1]+b[3])/2) + ' ' + str(b[2]-b[0]) + ' ' + str(b[3]-b[1]) + '\n'
            saving.write(saving_str)
    return UNSURE_LABEL_NUM, POS_LABEL_NUM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/custom/images", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_1024_48.pth",
                        help="path to weights file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names",
                        help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=2048, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = parse_data_config(opt.data_config)
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    num_unsure, num_pos = f(model, opt.img_size, opt.batch_size, data_config['valid'])
    print("Generated unsure labels: " + str(num_unsure) + " pos labels: " + str(num_pos))
    num_unsure, num_pos = f(model, opt.img_size, opt.batch_size, data_config['train'])
    print("Generated unsure labels: " + str(num_unsure) + " pos labels: " + str(num_pos))