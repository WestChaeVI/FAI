# 파이토치 데이터 클래스 생성
import os
import glob
import cv2
import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets

# resize는 transforms에서
class Dataset(Dataset):

    def __init__( self, root_path, mode, transforms=None):
        # root_path : '/home/sh/lab/eonc/skin_dataset/'    //    mode : train, valid, test
        self.all_images = sorted(glob.glob(os.path.join(root_path, mode, 'images', '*')))  
        self.all_labels = sorted(glob.glob(os.path.join(root_path, mode, 'labels', '*')))
        self.transforms = transforms


    def __getitem__(self, index):

        if torch.is_tensor(index):        # 인덱스가 tensor 형태일 수 있으니 리스트 형태로 바꿔준다.
            index = index.tolist()

        img_path = self.all_images[index]
        label_path = self.all_labels[index]


        # 이미지 전처리
        image_bgr = cv2.imread(img_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환
        image = image/255

        # 라벨 전처리
        mask = cv2.imread(label_path)[:,:,0]

        # label = np.concatenate([mask1, mask2], axis=2) # aixs=2에서 1번째 배열(물체:1, 배경:0)이 타겟
        image = torch.FloatTensor(np.transpose(image, [2, 0 ,1])) # Pytorch uses the channels in the first dimension
        mask = torch.LongTensor(mask).unsqueeze(0) # Adding channel dimension to label
        
        # transform 적용
        # sample = torch.cat((image, mask), 0) # insures that the same transform is applied
        if self.transforms != None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return image, mask        
        

        # # transform 적용
        # sample = torch.cat((image, mask), 0) # insures that the same transform is applied
        # if self.transforms != None:
        #     sample = self.transforms(sample)
        # image = sample[:image.shape[0], ...]
        # mask = sample[image.shape[0]:, ...]
        # return image, mask
        # # return sample

    def __len__(self):
        length = len(self.all_images)
        return length