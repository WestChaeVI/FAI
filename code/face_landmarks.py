import cv2
import numpy as np
import mediapipe as mp
import urllib.request

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# def face_landmarks(data_path):
#
#     # 이미지 파일의 경우을 사용하세요.:
#     IMAGE_FILES = [data_path]
#     drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#     with mp_face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5) as face_mesh:
#         for idx, file in enumerate(IMAGE_FILES):
#
#             with urllib.request.urlopen(file) as response:
#                 image_data = response.read()
#
#             # 이미지 데이터를 numpy 배열로 변환
#             image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
#             image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#             # 작업 전에 BGR 이미지를 RGB로 변환합니다.
#             results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     face_landmarks = results.multi_face_landmarks[0]
#
#     return face_landmarks
