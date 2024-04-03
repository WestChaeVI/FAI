import cv2
import torch
import numpy as np
from app.models import Result
from PIL import Image as PILImage
from settings.s3 import MyS3Client
from model.Unet.unet_models import unet
from code.image_processor import image_masking


def redness_analysis(folder_location, general_image, eval_image, face, device):
    
    s3_client = MyS3Client()

    model = unet(outchannels=1).to(device)
    model.load_state_dict(
        torch.load('weights/redness_Unet_292_0.469.pth', map_location=torch.device('cpu'))['net'])

    model.eval()
    with torch.no_grad():
        pred = model(eval_image)

    if isinstance(pred, dict):
        pred = pred['out']

    pred = torch.sigmoid(pred)

    mask = pred.clone()

    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    with torch.no_grad():
        mask = mask.squeeze()  # (2, 224, 224) 오류 남 (224, 224)가 되야함!
        # mask = torch.argmax( mask, dim=1 )  # argmax 하기전에는 (1, 2, 224, 224)


    # mask = mask.to(device = 'cpu', dtype = torch.int64).numpy() # tensor to numpy (반드시 디바이스도 변경)
    # 색 입히는 용도의 미리 마스킹 하는 부분
    mask = mask.cpu().detach().numpy()
    mask = np.stack((mask,) * 3, axis=-1)  # (224,224,3)

    # face 자른 거 224로 되돌리기
    resized_face = cv2.resize(face, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    resized_face = np.stack((resized_face,) * 3, axis=-1)

    color_mapping = {
        "R": 255,
        "G": 192,
        "B": 203
    }
    
    result_img = cv2.resize(general_image, (mask.shape[1], mask.shape[0]), interpolation = cv2.INTER_LINEAR)
    masked_image, mask_ = image_masking(color_mapping, result_img, mask, resized_face)

    masked_image = cv2.resize(masked_image, (general_image.shape[1], general_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    data = PILImage.fromarray(masked_image)
    image_link = s3_client.upload(folder=folder_location, file=data)

    redness_score = round(len(mask_[mask_ != 0]) / len(resized_face[resized_face != 0]) * 100, 2)

    return Result(
        score=redness_score,
        image=image_link,
        mask=mask_
    )        
