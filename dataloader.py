import torch
import os
from PIL import Image
import numpy as np
import scipy.signal as sig
import cv2


def get_image_list(raw_image_path, clear_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(clear_image_path + image_file), image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, None, image_file])
    return image_list


class RUE_Net_DataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.is_train = is_train
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path, is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        gray_img = np.array(raw_image.convert("L"))
        kx = np.array([[-1, 0, 1]])
        ky = np.array([[-1], [0], [1]])
        gx = np.zeros_like(gray_img)
        gy = np.zeros_like(gray_img)
        gx = ((sig.convolve2d(gray_img, kx, 'same') + 255)/2).astype('uint8')
        gy = ((sig.convolve2d(gray_img, ky, 'same') + 255)/2).astype('uint8')


        
        if self.is_train:
            clear_image = Image.open(clear_image)
            return self.transform(raw_image), self.transform(clear_image), self.transform(Image.fromarray(gray_img.astype('uint8'))), self.transform(Image.fromarray(gx)), self.transform(Image.fromarray(gy)), "_"
        return self.transform(raw_image), "_", self.transform(Image.fromarray(gray_img.astype('uint8'))), self.transform(Image.fromarray(gx)), self.transform(Image.fromarray(gy)), image_name

    def __len__(self):
        return len(self.image_list)
