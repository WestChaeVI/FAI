import cv2
import torch
import numpy as np
from app.models import Result
from PIL import Image as PILImage
from settings.s3 import MyS3Client
from model.Unet.unet_models import unet
from shapely import Point, Polygon
from ultralytics import YOLO


def dent_analysis(input_data_path, folder_location, general_image, boundary_points):
    
    s3_client = MyS3Client()
    
    all_boxes= []
    
    model = YOLO('/home/sh/lab/eonc/experiment/weights/yolov8_dent_best.pt') 

    results = model(input_data_path)  

    for result in results:
        boxes= result.boxes  

    # Process results list
    for result in results:
        boxes= result.boxes
        cls = boxes.cls
        xywh = boxes.xywh
        for i in range(len(cls)):
            if cls[i] == 0 : 
                boundary_polygon = Polygon(boundary_points)
                point = Point(xywh[i][0].detach().cpu(), xywh[i][1].detach().cpu())
                if boundary_polygon.contains(point):
                    all_boxes.append((int(xywh[i][0]), int(xywh[i][1])))
                    
    points = all_boxes

    points_list = []

    if len(points)==0 :
        dent_img = general_image
        dent_score = 0.0

    else:
        for point in points:
            (h,w) = point
            dent_img = cv2.circle(general_image, (h,w), 20, (49, 5, 39), 4)
            points_list.append(point)
            dent_score = float(len(points_list))

    data = PILImage.fromarray(dent_img)
    s3_client.upload(folder=folder_location, file=data)

    return Result(score = dent_score,
                  image = folder_location,
                  mask = points_list)
