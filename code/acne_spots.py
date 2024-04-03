import cv2
import urllib
import numpy as np
import torch
import matplotlib.pyplot as plt
from shapely import Point, Polygon

from ultralytics import YOLO


def is_point_inside_boundary(x, y, boundary_points):
    # 주어진 점이 바운더리 안에 있는지 확인
    # 바운더리 안에 있는지 확인하기 위해 점이 다각형 내부에 있는지 확인하는 알고리즘 사용 가능
    # 여기서는 단순히 바운더리의 경계점을 이용하여 점이 그 안에 있는지 확인하는 방식 사용
    boundary_polygon = Polygon(boundary_points)
    point = Point(x, y)
    return boundary_polygon.contains(point)
