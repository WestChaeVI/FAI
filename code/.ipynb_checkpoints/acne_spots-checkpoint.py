import cv2
import urllib
import numpy as np
import torch
import matplotlib.pyplot as plt
from shapely import Point, Polygon

from ultralytics import YOLO


def oil_segment(data_path):
    
    
    model = YOLO('/home/sh/lab/eonc/experiment/weights/yolov8_seg_oil_best.pt') 

    results = model(data_path) 
    
    for result in results:
        masks = result.masks
        
    return masks


def is_point_inside_boundary(x, y, boundary_points):
    # 주어진 점이 바운더리 안에 있는지 확인
    # 바운더리 안에 있는지 확인하기 위해 점이 다각형 내부에 있는지 확인하는 알고리즘 사용 가능
    # 여기서는 단순히 바운더리의 경계점을 이용하여 점이 그 안에 있는지 확인하는 방식 사용
    boundary_polygon = Polygon(boundary_points)
    point = Point(x, y)
    return boundary_polygon.contains(point)


def acne_spot(data_path, boundary_points):
    
    all_boxes= []
    
    model = YOLO('/home/sh/lab/eonc/experiment/weights/yolov8_acne_best.pt') 

    results = model(data_path)  

    for result in results:
        boxes= result.boxes  

    points = []
    for i, box in enumerate(boxes):
        x = int(box.xywh[0][0].item())
        y = int(box.xywh[0][1].item())
        
        if is_point_inside_boundary(x,y, boundary_points):
            all_boxes.append((int(x),int(y)))

    return all_boxes


# 위 함수를 이용하여 여드름을 detect한 결과 중 바운더리 안에 있는 박스들의 위치 정보만 가져오기
def dent_spot(data_path, boundary_points):
    
    all_boxes= []
    
    model = YOLO('/home/sh/lab/eonc/experiment/weights/yolov8_dent_best.pt') 

    results = model(data_path)  

    for result in results:
        boxes= result.boxes  

    # Process results list
    for result in results:
        boxes= result.boxes
        cls = boxes.cls
        xywh = boxes.xywh
        for i in range(len(cls)):
            if cls[i] == 0 :                
                if is_point_inside_boundary(xywh[i][0].detach().cpu(), xywh[i][1].detach().cpu(), boundary_points):
                    all_boxes.append((int(xywh[i][0]), int(xywh[i][1])))
    

    return all_boxes


def eyebag_segment(data_path):
    
    model = YOLO('/home/sh/lab/eonc/experiment/weights/yolov8_seg_eyebag_best.pt') 

    results = model(data_path) 
    
    # results가 비어 있는 경우
    if not results[0].masks:
        print("No detections found.")
        return None
    
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

        return summed_mask

def eczema_segment(data_path):
    
    mask = []
    
    model = YOLO('/home/sh/lab/eonc/experiment/weights/yolov8_seg_eczema_best.pt') 

    results = model(data_path) 
    
    # results가 비어 있는 경우
    if not results[0].masks:
        print("No detections found.")
        return None
    
    else:
        # results가 있는 경우
        summed_mask = None
        for result in results:
            masks = result.masks
            # Initialize an empty tensor to store the summed mask
            summed_mask = torch.zeros_like(masks[0].data)
            # Sum all masks
            for mask in masks:
                summed_mask += mask.data
            # Set non-zero values to 1
            summed_mask[summed_mask != 0] = 1
            # Assign the modified summed mask back to all masks
            for i in range(len(masks)):
                masks[i].data = summed_mask.clone()

        return summed_mask
