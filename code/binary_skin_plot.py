# Library improt
import os
import copy
import glob
import cv2
import random
import uuid
import datetime
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from PIL import Image as PILImage

import mimetypes
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.utils as vutils
from torchvision import transforms, datasets

# Dataset improt
from dataset import Dataset
from data_loader import Data_loader

# Model import
from model.Unet.unet_models import UNet, unet
from model.Unet.unet_parts import *
from model.Deeplab.Deeplabv3 import deeplabv3_resnet101

# Metrics import
from code.metrics import DiceCELoss ,dice_score ,iou ,dice_pytorch_eval, iou_pytorch_eval, DiceBCELoss
from code.metrics import DiceLossMulticlass
from code.metrics import mIOU

from code.ploting import plot_model_prediction, plot_mode_prediction

from code.skin_texture import skin_texture_plot

from pydantic import BaseModel
from app.models import ResponseModel

from settings.s3 import MyS3Client

# cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
        
        
        
def binary_skin_plot(email, input_data_path, device=device):
    '''
    0 - Background
    1 - Acne [빨강]
    2 - Eye_bags [파랑]
    3 - Hyperpigmentation(과다색소침착) [핑크]
    4 - black_head [보라색]
    5 - wrinkle [민트]
    6 - face
    '''

    # response = ResponseModel()
    s3_client = MyS3Client()

    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    
    # AWS 서버 경로 만들기 -----------------------------------------------
    field = 'pid_{}/{}/{}/{}/'.format('tjcowns@gmail.com', year, month, day)

    ance_image_path = field + 'imageance.' + str(uuid.uuid4()) + '.png'
    eyebags_image_path = field + 'imageeyebags.' + str(uuid.uuid4()) + '.png'
    hyperpig_image_path = field + 'imagehyperpig.' + str(uuid.uuid4()) + '.png'
    blackhead_image_path = field + 'imageblackhead.' + str(uuid.uuid4()) + '.png'
    wrinkle_image_path = field + 'imagewrinkle.' + str(uuid.uuid4()) + '.png'
    overall_image_path = field + 'imageoverall.' + str(uuid.uuid4()) + '.png'
    # --------------------------------------------------------------------
    
    acne_model = unet(outchannels=1).to(device)
    acne_model.load_state_dict(torch.load('/home/sh/lab/eonc/check_points/Acne_Unet_397_0.46.pth')['net'])

    eyebags_model = unet(outchannels=1).to(device)
    eyebags_model.load_state_dict(torch.load('/home/sh/lab/eonc/check_points/Eye_bags_Unet_400_0.0.pth')['net'])

    hyperpigment_model = unet(outchannels=1).to(device)
    hyperpigment_model.load_state_dict(torch.load('/home/sh/lab/eonc/check_points/Hyperpigment_Unet_111_0.235.pth')['net'])

    blackhead_model = unet(outchannels=1).to(device)
    blackhead_model.load_state_dict(torch.load('/home/sh/lab/eonc/check_points/Blackhead_Unet_400_0.0.pth')['net'])

    wrinkle_model = unet(outchannels=1).to(device)
    wrinkle_model.load_state_dict(torch.load('/home/sh/lab/eonc/check_points/Wrinkle_Unet_266_0.459.pth')['net'])


    size = (224, 224)

    # eval 전 이미지 전처리
    img = cv2.imread(input_data_path)
    (H,W,C) = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)

    result_img = img.copy()


    eval_image = img / 255.0
    eval_image = eval_image.astype(np.float32)
    eval_image = eval_image.transpose((2,0,1))
    eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
    eval_image = eval_image.to( device=device, dtype = torch.float32 )


    model_list = [acne_model, eyebags_model, hyperpigment_model, blackhead_model, wrinkle_model]
    mask_list = []

    for i, model in enumerate(model_list) :
        # we do not need to calculate gradients
        model.eval()
        with torch.no_grad():
            # Prediction
            pred = model(eval_image)  # (1, 7, 224, 224)


        if isinstance(pred, dict):
            pred = pred['out']  # (1 7 224 224)

        pred = torch.sigmoid(pred)  # (1 7 224 224)

        mask = pred.clone()

        # 0.5를 기준으로 마스크 만들기.
        mask[mask >= 0.5 ] = 1
        mask[mask < 0.5 ] = 0
        with torch.no_grad(): 
        #   mask = torch.argmax( mask, dim=1 )  # argmax 하기전에는 (1, 2, 224, 224)
            mask = mask.squeeze() # (2, 224, 224) 오류 남 (224, 224)가 되야함!

        if i == 1:
            mask[mask == 1] = 2
        if i == 2:
            mask[mask == 1] = 3
        if i == 3:
            mask[mask == 1] = 4
        if i == 4:
            mask[mask == 1] = 5

        # mask = mask.to(device = 'cpu', dtype = torch.int64).numpy() # tensor to numpy (반드시 디바이스도 변경)
        mask = mask.cpu().detach().numpy()
        mask = np.stack( (mask,)*3, axis=-1 ) # (224,224,3)

        mask_list.append(mask)

    result_img = img.copy()
    result_img_2 = img.copy()

    color_mapping = {
    1: (255, 0, 0),   # 빨강색
    2: (0, 0, 255),   # 파랑색
    3: (255, 192, 203),   # 핑크색
    4: (128, 0, 128),   # 보라색
    5: (0, 255, 128)   # 민트색
    }

    for k, mask in enumerate(mask_list) :
        if k == 0 :
            continue

        result_img[mask == (i+1)] = 0
        reslut_mask = mask.copy()
        reslut_mask_0, reslut_mask_1, reslut_mask_2 = reslut_mask[:,:,0], reslut_mask[:,:,1], reslut_mask[:,:,2]
        reslut_mask_0[reslut_mask_0 == (k+1)] = color_mapping[k+1][0]
        reslut_mask_1[reslut_mask_1 == (k+1)] = color_mapping[k+1][1]
        reslut_mask_2[reslut_mask_2 == (k+1)] = color_mapping[k+1][2]

        result_img += reslut_mask.astype(np.uint8)

        objective_img = result_img_2 + reslut_mask.astype(np.uint8)   # only one object

        if k == 1 :
            objective_img = cv2.resize(objective_img, (W,H), interpolation = cv2.INTER_LINEAR)
            data1 = PILImage.fromarray(objective_img)
            s3_client.upload(folder=eyebags_image_path, file=data1)
        elif k == 2 :
            objective_img = cv2.resize(objective_img, (W,H), interpolation = cv2.INTER_LINEAR)
            data2 = PILImage.fromarray(objective_img)
            s3_client.upload(folder=hyperpig_image_path, file=data2)
        elif k == 3 :
            objective_img = cv2.resize(objective_img, (W,H), interpolation = cv2.INTER_LINEAR)
            data3 = PILImage.fromarray(objective_img)
            s3_client.upload(folder=blackhead_image_path, file=data3) 
        elif k == 4 :
            objective_img = cv2.resize(objective_img, (W,H), interpolation = cv2.INTER_LINEAR)
            data4 = PILImage.fromarray(objective_img)
            s3_client.upload(folder=wrinkle_image_path, file=data4)              
    # score
    (h,w) = size
    ance_score = round(len(mask_list[0][mask_list[0] != 0]) / (h*w) * 100, 2)
    
    eyebags_score = round(len(mask_list[1][mask_list[1] != 0]) / (h*w) * 100, 2)
    
    hyperpigment_score = round(len(mask_list[2][mask_list[2] != 0]) / (h*w) * 100, 2)
    
    blackhead_score = round(len(mask_list[3][mask_list[3] != 0]) / (h*w) * 100, 2)
    
    wrinkle_score = round(len(mask_list[4][mask_list[4] != 0]) / (h*w) * 100, 2)
    
    print(f'Acne : {ance_score}%\nEye bags : {eyebags_score}%\nHyperpigment : {hyperpigment_score}%\nBlack head : {blackhead_score}%\nWrinkle : {wrinkle_score}%\n')
    
    result_data = PILImage.fromarray(result_img)
    
    # skin texture
    texture_image_data = skin_texture_plot(data_path=input_data_path, result_data=result_data, color=(255,105,180), thickness=4)
    texture_img = np.array(texture_image_data)
    
    img = cv2.imread(input_data_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # ance spots
    points = acne_spot(input_data_path)
    
    for point in points:
        (h,w) = point
        texture_img = cv2.circle(texture_img, (h*2,w*2), 20, (0, 0, 0), 4)

        ance_img = cv2.circle(img, (h,w), 60, (0, 0, 0), 4)
    
    data5 = PILImage.fromarray(ance_img)
    s3_client.upload(folder=ance_image_path, file=data5)     
    
    data6 = PILImage.fromarray(texture_img[:,:,::-1])
    s3_client.upload(folder=overall_image_path, file=data6)
    
    print("Building response model...")

    response = ResponseModel(
    acne=ance_score,
    acne_image=ance_image_path,
    age_spot=eyebags_score,
    age_spot_image=eyebags_image_path,
    redness=hyperpigment_score,
    redness_image=hyperpig_image_path,
    texture=blackhead_score,
    texture_image=blackhead_image_path,
    wrinkle=wrinkle_score,
    wrinkle_image=wrinkle_image_path,
    overall_image=overall_image_path)

    print("Finish building response model")

    return response
