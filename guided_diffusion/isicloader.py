import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from torchvision.utils import save_image
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"
label_suffix = ".png"

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name) #.replace('.jpg', label_suffix)


class ISICDataset(Dataset):
    # def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):
    def __init__(self, args, data_path, transform = None, mode = 'train', plane=False):


        self.split = "test"
        self.root_dir = "/media/lscsc/nas/yihan/ddpm_1/MedSegDiff/Levier"
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        # if self.data_len <= 0:
        #     self.data_len = self.dataset_len
        # else:
        self.data_len = self.dataset_len

        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        """Get the images"""

        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A   = Image.open(A_path).convert("RGB")
        img_B   = Image.open(B_path).convert("RGB")

        L_path  = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        img_lbl = Image.open(L_path).convert("L")

        # name = self.name_list[index]+'.jpg'
        # img_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',name)
        
        # mask_name = name.split('.')[0] + '_Segmentation.png'
        # msk_path = os.path.join(self.data_path, 'ISBI2016_ISIC_Part3B_'+ self.mode +'_Data',mask_name)

        # img = Image.open(img_path).convert('RGB')
        # mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            torch.set_rng_state(state)
            img_lbl = self.transform(img_lbl)
        name = self.img_name_list[index % self.data_len]
     
        # if self.mode == 'Training':
        return (img_A, img_B, img_lbl)
        # return (img_A, img_B, img_lbl, self.img_name_list[index % self.data_len])
        # else:
        #     return (img, mask, name)