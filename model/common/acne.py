import cv2
import torch
import numpy as np
from app.models import Result
from PIL import Image as PILImage
from settings.s3 import MyS3Client
from model.Unet.unet_models import unet
from code.utils import get_filename
from shapely import Point, Polygon
from ultralytics import YOLO

# cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
    
    

def acne_analysis(device: str):
    model = unet(outchannels=1).to(device)
    model.load_state_dict(
        torch.load('weights/Acne_Unet_397_0.46.pth', map_location=torch.device('cpu'))['net'])

    # model.eval()은 주로 테스트 데이터나 검증 데이터를 사용하여 모델을 평가할 때 사용됩니다.
    # 평가 모드에서는 모델이 추론 시에 동일한 동작을 수행하도록 설정되어 있어, 모델의 성능 평가에 불필요한 노이즈를 줄이고 일관된 결과를 얻을 수 있습니다.
    # 모델을 학습(training)하는 동안 사용한 모델 객체를 추론(inference)할 때 model.eval()을 호출하여 추론 모드로 전환하고,
    # 추론이 끝난 후에는 다시 model.train()을 호출하여 학습 모드로 전환하는 것이 일반적입니다.
    model.eval()
    with torch.no_grad():
        pred = model(image)

    if isinstance(pred, dict):
        pred = pred['out']

    pred = torch.sigmoid(pred)

    mask = pred.clone()

    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    with torch.no_grad():
        mask = mask.squeeze()  # (2, 224, 224) 오류 남 (224, 224)가 되야함!
        # mask = torch.argmax( mask, dim=1 )  # argmax 하기전에는 (1, 2, 224, 224)




def acne_detect_analysis(input_data_path, folder_location, general_image, boundary_points):
    
    s3_client = MyS3Client()
    
    all_boxes= []
    
    model = YOLO('weights/yolov8_acne_best.pt')

    results = model(input_data_path)  

    for result in results:
        boxes= result.boxes  

    for i, box in enumerate(boxes):
        x = int(box.xywh[0][0].item())
        y = int(box.xywh[0][1].item())
        
        boundary_polygon = Polygon(boundary_points)
        point = Point(x, y)
        
        if boundary_polygon.contains(point):
            all_boxes.append((int(x),int(y)))
            
    points = all_boxes

    points_list = []

    if len(points)==0 :
        acne_img = general_image
        acne_score = 0.0

    else:
        for point in points:
            (h,w) = point
            acne_img = cv2.circle(general_image, (h,w), 20, (0, 0, 0), 4)
            points_list.append(point)
            acne_score = float(len(points_list))

    data = PILImage.fromarray(acne_img)
    s3_client.upload(folder=folder_location, file=data)

    return Result(score = acne_score,
                  image = folder_location,
                  mask = points_list)
