import torch
import torch.nn.functional as F
import random
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageFilter
from torchvision import transforms

def random_augmentation(images, targets):
    if np.random.random() < 0.5:
        images, targets = fun_bright(images, targets)
    if np.random.random():
        images, targets = fun_color(images, targets)
    if np.random.random() < 0.5:
        x = np.random.random()
        if x <= 0.4:
            images, targets = gaussian_blur(images, targets)
        elif x > 0.4 and x <= 0.6:
            images, targets = gaussian_noise(images, targets)
        else:
            images, targets = fun_Sharpness(images, targets)

    if np.random.random() < 0.5:
        images, targets = horisontal_flip(images, targets)
    if np.random.random() < 0.5:
        images, targets = fun_Contrast(images, targets)

    return images, targets

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def gaussian_noise(images, targets):
    noise = torch.randn_like(images[0, :, :])/255
    noise = noise.unsqueeze(0).repeat(3, 1, 1)
    return images + noise, targets

def gaussian_blur(images, targets):

    kernel_size = random.uniform(0, 1.5)
    images = transforms.ToPILImage(mode='RGB')(images)
    images = images.filter(ImageFilter.GaussianBlur(radius=kernel_size))
    # images = ImageFilter.GaussianBlur(radius=kernel_size[0]).filter(transforms.ToPILImage(mode='RGB')(images))
    # images = cv2.GaussianBlur(images.numpy(), (kernel_size[0], kernel_size[0]), 0)
    # return torch.from_numpy(images), targets
    return transforms.ToTensor()(images), targets

def fun_color(image, target):
    # 色度,增强因子为1.0是原始图像
    # 色度增强 1.5
    # 色度减弱 0.8
    image = transforms.ToPILImage(mode='RGB')(image)
    coefficient = random.uniform(0.5, 1.5)
    enh_col = ImageEnhance.Color(image)
    image_colored1 = enh_col.enhance(coefficient)
    return transforms.ToTensor()(image_colored1), target

def fun_Contrast(image, target):
    # 对比度，增强因子为1.0是原始图片
    # 对比度增强 1.5
    # 对比度减弱 0.8
    image = transforms.ToPILImage(mode='RGB')(image)
    coefficient = random.uniform(0.5, 1.5)
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted1 = enh_con.enhance(coefficient)
    return transforms.ToTensor()(image_contrasted1), target

def fun_Sharpness(image, target):
    # 锐度，增强因子为1.0是原始图片
    # 锐度增强 3
    # 锐度减弱 0.8
    image = transforms.ToPILImage(mode='RGB')(image)
    coefficient = random.uniform(1.5, 3)
    enh_sha = ImageEnhance.Sharpness(image)
    image_sharped1 = enh_sha.enhance(coefficient)
    return transforms.ToTensor()(image_sharped1), target

def fun_bright(image, target):
    # 变亮 1.5
    # 变暗 0.8
    # 亮度增强,增强因子为0.0将产生黑色图像； 为1.0将保持原始图像。
    image = transforms.ToPILImage(mode='RGB')(image)
    coefficient = random.uniform(0.5, 1.5)
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened1 = enh_bri.enhance(coefficient)
    return transforms.ToTensor()(image_brightened1), target