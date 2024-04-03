import cv2
import torch
import numpy as np
from app.models import Result
from PIL import Image as PILImage
from settings.s3 import MyS3Client
from model.Unet.unet_models import unet
from ultralytics import YOLO


def eyebag_analysis(input_data_path, folder_location,  general_image, face, alpha = 0.5):
    
    s3_client = MyS3Client()
    
    H,W,C = general_image.shape
    
    model = YOLO('/home/sh/lab/eonc/experiment/weights/yolov8_seg_eyebag_best.pt') 

    results = model(input_data_path) 
    
    # results가 비어 있는 경우
    if not results[0].masks:
        print("No detections found.")
    
    else:
        # results가 있는 경우
        summed_mask = None
        for result in results:
            masks = result.masks
            # Initialize an empty tensor to store the summed mask
            summed_mask = torch.zeros_like(masks[0].data)
            # Sum all masks
            for mask in masks[:2]:
                summed_mask += mask.data
            # Set non-zero values to 1
            summed_mask[summed_mask != 0] = 1
            # Assign the modified summed mask back to all masks
            for i in range(len(masks)):
                masks[i].data = summed_mask.clone()

    mask = summed_mask
    
    if mask is None:
        eyebag_score = 0.0
        eyebag_image = general_image
        mask_1 = None

    else:
        mask = mask.detach().cpu().numpy().transpose(1, 2, 0)
        mask = mask.squeeze(2)

        color = (128 / 255, 128 / 255, 128 / 255)

        resized_face = cv2.resize(face, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)  # (640, 576)
        resized_face[resized_face != 0] = 1

        mask_1 = mask * resized_face

        eyebag_score = round(len(mask_1[mask_1 != 0]) / len(resized_face[resized_face != 0]) * 100, 2)

        general_image = cv2.resize(general_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        for c in range(3):
            general_image[:, :, c] = np.where(mask_1 == 1,
                                        general_image[:, :, c] *
                                        (1 - alpha) + alpha * color[c] * 255,
                                        general_image[:, :, c])

        eyebag_image = cv2.resize(general_image, (W, H), interpolation=cv2.INTER_LINEAR)

    data = PILImage.fromarray(eyebag_image)
    s3_client.upload(folder=folder_location, file=data)
    
    return Result(score = eyebag_score,
                  image = folder_location,
                  mask = mask_1) # (640, 640) 
