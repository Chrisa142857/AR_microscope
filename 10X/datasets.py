import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from datetime import datetime
from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size):
    image = transforms.ToTensor()(transforms.Resize(size)(transforms.ToPILImage()(image)))
    # image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, reg=False, img_size=(416, 416)):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        self.reg = reg

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        # img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution

        # Extract image as PyTorch tensor
        img = Image.open(img_path).convert('RGB')
        # img = self.img
        # h, w = img.size
        # resize_scale = self.img_size / max(h, w)
        # img = img.resize((int(h*resize_scale), int(w*resize_scale)))
        if self.reg:
            img = Image.fromarray(normalize(np.array(img)).astype('uint8'),'RGB')
        # img = transforms.Grayscale()(img)
        img = transforms.ToTensor()(img)
        # if not self.reg:
            # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            # img = transforms.Normalize(mean=[0.449], std=[0.226])(img)

        # img, _ = pad_to_square(img, 0)
        # Resize
        # img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img, img_size=(416, 416), with_unsure=False, augment=True, reg=True, multiscale=True, normalized_labels=True, use_pad=False):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        if with_unsure:
            self.label_files = [
                path.replace("images", "labels").replace(".png", "_withUnsure.txt").replace(".jpg", "_withUnsure.txt")
                for path in self.img_files
            ]
        else:
            self.label_files = [
                path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        # self.min_size = self.img_size - 3 * 32
        # self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.img = img
        self.reg = reg
        self.use_pad = use_pad

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = Image.open(img_path).convert('RGB')
        # img = self.img
        # h, w = img.size
        # # resize_scale = self.img_size / max(h, w)
        # img = img.resize(self.img_size[0], self.img_size[1])
        # img = img.resize((int(h*resize_scale), int(w*resize_scale)))
        if self.use_pad:
            img = img.resize((self.img_size[1]+64, self.img_size[1]))
        if self.reg:
            img = Image.fromarray(normalize(np.array(img)).astype('uint8'),'RGB')
        # img = transforms.Grayscale()(img)
        img = transforms.ToTensor()(img)
        # if not self.reg:
        #     img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        #     # img = transforms.Normalize(mean=[0.449], std=[0.226])(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        # img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            # x1 += pad[0]
            # y1 += pad[2]
            # x2 += pad[1]
            # y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            boxes[:, 1] = (((x1 + x2) / 2) / padded_w)
            boxes[:, 2] = (((y1 + y2) / 2) / padded_h)

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes




        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img.unsqueeze(0), targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        # if self.multiscale and self.batch_count % 10 == 0:
        #     self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.cat(imgs)
        # imgs = torch.stack([img for img in imgs])
        # imgs = F.interpolate(imgs, size=self.img_size, mode="nearest")
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
