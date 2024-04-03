import cv2
import mediapipe as mp
from IPython.display import Image, display
import numpy as np
import itertools
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from code.image_processor import ImageProcessor


def skin_texture_plot(data_path, result_data, color=(255,105,180), thickness=3):

    # MediaPipe FaceMesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    imgproc = ImageProcessor(data_path)
    general_image, _, resized_general_image = imgproc.image_preprocess()
    h,w,c = general_image.shape

    result_image = np.array(result_data)
    result_image = cv2.resize(result_image, (w,h), interpolation=cv2.INTER_LINEAR)  # 이미지를 2배로 확대

    # 얼굴 특성 추출
    results = face_mesh.process(general_image)

    mark_dict = {'upper nose': [168],
                 'left route': [193, 244, 233, 232, 231, 230, 229, 228, 31, 35, 143, 234, 93, 215, 138, 214, 212, 216, 206, 203],
                 'center nose': [4],
                 'right route': [423, 426, 436, 432, 434, 367, 435, 323, 454, 372, 265, 261, 448, 449, 450, 451, 452, 453, 464, 417]}

    order = ['upper nose', 'left route', 'center nose', 'right route']

    all_landmarks = []
    image_landmarks = imgproc.face_landmarks()
    
    # 랜드마크 그리기 및 번호 표시
    if results.multi_face_landmarks:
        for key in order:
            value = mark_dict[key]
            for i in range(len(value)):
                idx = value[i]

                x = int(image_landmarks.landmark[idx].x * resized_general_image.shape[1])
                y = int(image_landmarks.landmark[idx].y * resized_general_image.shape[0])

                # 랜드마크를 저장
                all_landmarks.append((x, y))

    # 랜드마크를 연결하는 선 그리기
    for i in range(len(all_landmarks)):
        x1, y1 = all_landmarks[i]
        x2, y2 = all_landmarks[(i + 1) % len(all_landmarks)]
        cv2.line(result_image, (x1, y1), (x2, y2), color, thickness)
    
    result_image_data = PILImage.fromarray(result_image)
    
    return result_image_data



def face_recognize(data_path) :
    
    # MediaPipe FaceMesh 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
        # eval 전 이미지 전처리
    with urllib.request.urlopen(data_path) as response:
        image_data = response.read()

    image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    image_landmarks = get_face_landmarks(data_path)
    mark = [10, 338, 297, 332, 284, 251, 389, 356, 447, 366, 435, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 215, 137, 234, 127, 139, 21, 54, 103, 67, 109]

    all_landmarks = []

    # 랜드마크 그리기 및 번호 표시
    for idx in mark:
        x = int(image_landmarks.landmark[idx].x * image.shape[1])
        y = int(image_landmarks.landmark[idx].y * image.shape[0])

        # 랜드마크를 저장
        all_landmarks.append((x, y))

    # 랜드마크를 연결하는 선 그리기
    for i in range(len(all_landmarks)):
        x1, y1 = all_landmarks[i]
        x2, y2 = all_landmarks[(i + 1) % len(all_landmarks)]
        cv2.line(image, (x1, y1), (x2, y2), (255, 105, 180), 3)

    # 다각형 그리기
    points = np.array(all_landmarks, np.int32)
    cv2.fillPoly(image, [points], (255, 105, 180))  # 테두리 안을 색으로 채움

    # 검정색 배경의 이미지 생성
    black_background = np.zeros_like(image)

    # 흰색으로 채워진 다각형 영역 추가
    cv2.fillPoly(black_background, [points], (255, 255, 255))

    # 흑백 이미지로 변환
    binary_image = cv2.cvtColor(black_background, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    return binary_image, all_landmarks