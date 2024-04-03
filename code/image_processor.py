import cv2
import torch
import urllib.request
import numpy as np
import mediapipe as mp
from IPython.display import Image, display
from PIL import Image as PILImage

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)

    
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class ImageProcessor:
    image_array: np.ndarray

    def __init__(self, image_path):
        with urllib.request.urlopen(image_path) as response:
            image_data = response.read()

        self.image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
        
        self.device = device

    def image_preprocess(self):
        # eval_image -> pre_segment_image, segmentation model에 입력될 이미지
        # general_image -> pre_detect_image, detection model에 입력될 이미지
        # result_img -> pre_overall_image, 최종 overall_image를 기록할 이미지

        # eval 전 이미지 전처리
        size = (224, 224)  # 성능이 더 나아질때 숫자를 올리기. 256만 되어도 고성능의 연산을 필요로 한다.
        print(f"\n이미지 전처리 사이즈: {size}\n")

        image = cv2.imdecode(self.image_array, cv2.IMREAD_COLOR)
        (H, W, C) = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        general_image = image.copy()
        resized_general_image = image.copy()
        
        resized_general_image = cv2.resize(resized_general_image, (224,224), interpolation = cv2.INTER_LINEAR)

        # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

        eval_image = image / 255.0
        eval_image = eval_image.astype(np.float32)
        eval_image = eval_image.transpose((2, 0, 1))
        eval_image = torch.from_numpy(eval_image).unsqueeze(0)  # Batch 채널 추가 -> (1, 3, 256, 256)
        eval_image = eval_image.to(device=self.device, dtype=torch.float32)

        return general_image, eval_image, resized_general_image

    def face_landmarks(self):
        # 이미지 파일의 경우을 사용하세요.:
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            image = cv2.imdecode(self.image_array, cv2.IMREAD_COLOR)
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        face_landmarks = results.multi_face_landmarks[0]

        return face_landmarks

    def face_recognize(self):
        # MediaPipe FaceMesh 초기화
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        image = cv2.imdecode(self.image_array, cv2.IMREAD_COLOR)

        image_landmarks = self.face_landmarks()
        mark = [10, 338, 297, 332, 284, 251, 389, 356, 447, 366, 435, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
                136, 172, 215, 137, 234, 127, 139, 21, 54, 103, 67, 109]

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


def image_masking(color, target_image, mask, resized_face):

    target_image[mask == 1] = 0
    result_mask = mask.copy()
    result_mask_0, result_mask_1, result_mask_2 = result_mask[:, :, 0], result_mask[:, :, 1], result_mask[:, :, 2]
    result_mask_0[result_mask_0 == 1] = color["R"]
    result_mask_1[result_mask_1 == 1] = color["G"]
    result_mask_2[result_mask_2 == 1] = color["B"]
    
    mask_ = result_mask.astype(np.uint8) * resized_face
    
    target_image += mask_

    return target_image, mask_
