import cv2
import torch
import numpy as np
from app.models import Result
from PIL import Image as PILImage
from settings.s3 import MyS3Client
from ultralytics import YOLO


def oil_analysis(input_data_path, folder_location,  general_image, face, alpha = 0.5):
    
    s3_client = MyS3Client()
    
    H,W,C = general_image.shape
    
    model = YOLO('weights/yolov8_seg_oil_best.pt')

    results = model(input_data_path) 
    
    for result in results:
        masks = result.masks
        

    if masks is None:
        oil_score = 0.0
        oil_image = general_image
        mask_1 = None

    else:
        mask = masks[0].data
        mask = mask.detach().cpu().numpy().transpose(1, 2, 0)
        mask = mask.squeeze(2)

        color = (135 / 255, 206 / 255, 235 / 255)

        resized_face = cv2.resize(face, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)  # (640, 576)
        resized_face[resized_face != 0] = 1

        mask_1 = mask * resized_face

        oil_score = round(len(mask_1[mask_1 != 0]) / len(resized_face[resized_face != 0]) * 100, 2)

        general_image = cv2.resize(general_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        for c in range(3):
            general_image[:, :, c] = np.where(mask_1 == 1,
                                        general_image[:, :, c] *
                                        (1 - alpha) + alpha * color[c] * 255,
                                        general_image[:, :, c])

        oil_image = cv2.resize(general_image, (W, H), interpolation=cv2.INTER_LINEAR)

    data = PILImage.fromarray(oil_image)
    s3_client.upload(folder=folder_location, file=data)
    
    return Result(score = oil_score,
                  image = folder_location,
                  mask = mask_1) # (640, 640) 
