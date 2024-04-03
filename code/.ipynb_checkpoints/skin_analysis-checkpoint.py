# Library import
import cv2
import torch
import numpy as np
from PIL import Image as PILImage
from code.utils import get_filename
from code.skin_texture import skin_texture_plot
from code.yolo_masking import yolo_masking

# Model import
from code.image_processor import ImageProcessor, image_masking
from model.common.acne import acne_analysis, acne_detect_analysis
from model.common.age_spot import age_spot_analysis
from model.common.dent import dent_analysis
from model.common.eczema import eczema_analysis
from model.common.eyebag import eyebag_analysis
from model.common.oil import oil_analysis
from model.common.redness import redness_analysis
from model.common.texture import texture_analysis
from model.common.wrinkle import wrinkle_analysis
from app.models import ResponseModel
from settings.s3 import MyS3Client

# cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)


def skin_analysis(email, input_data_path, device=device):
    
    s3_client = MyS3Client()
    
    # 응답 데이터 초기화
    response = ResponseModel.from_email(email=email)

    image_processor = ImageProcessor(input_data_path)
    
    # 입력 이미지 전처리
    general_image, eval_image, resized_general_image = image_processor.image_preprocess()
    H, W, C = general_image.shape

    # 얼굴 경계면 인식
    face, boundary_points = image_processor.face_recognize()  # face mask

    results = []

    age_spot_result = age_spot_analysis(folder_location=response.age_spot_image, general_image=general_image, eval_image=eval_image, face=face, device=device)
    response.age_spot_image = age_spot_result.image
    results.append(age_spot_result)

    redness_result = redness_analysis(folder_location=response.redness_image, general_image=general_image, eval_image=eval_image, face=face, device=device)
    response.redness_image = redness_result.image
    results.append(redness_result)

    texture_result = texture_analysis(folder_location=response.texture_image, general_image=general_image, eval_image=eval_image, face=face, device=device)
    response.texture_image = texture_result.image
    results.append(texture_result)

    wrinkle_result = wrinkle_analysis(folder_location=response.wrinkle_image, general_image=general_image, eval_image=eval_image, face=face, device=device)
    response.wrinkle_image = wrinkle_result.image
    results.append(wrinkle_result)

    for result in results:
        mask = result.mask
        resized_general_image += mask   # (224, 224, 3)

    # 순서 :  oil -> eyebag -> eczema -> acne -> dent
    oil_result = oil_analysis(input_data_path, folder_location=response.oil_image,  general_image=general_image, face=face) 
    eyebag_result = eyebag_analysis(input_data_path, folder_location=response.eyebag_image,  general_image=general_image, face=face)
    eczema_result = eczema_analysis(input_data_path, folder_location=response.eczema_image,  general_image=general_image, face=face)
    acne_result = acne_detect_analysis(input_data_path, folder_location=response.acne_image, general_image=general_image, boundary_points=boundary_points) 
    dent_result = dent_analysis(input_data_path, folder_location=response.dent_image, general_image=general_image, boundary_points=boundary_points)


    # 투명도 설정
    alpha = 0.5
    oil_color = (135/255,206/255,235/255)
    eyebag_color = (128/255,128/255,128/255)
    eczema_color = (255/255,10/255,100/255)
    
    
    oil_image = yolo_masking(resized_general_image, oil_result, color=oil_color, alpha=0.5)
    eyebag_image = yolo_masking(oil_image , eyebag_result, color=eyebag_color, alpha=0.5)
    eczema_image = yolo_masking(resized_general_image, eczema_result, color=eczema_color, alpha=0.5)
    
    
#     if oil_result.mask is None :
#         oil_image = resized_general_image
        
#     else:
#         mask = oil_result.mask
        
#         image = resized_general_image.copy()
#         image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
#         for c in range(3):
#             image[:,:,c] = np.where(mask==1,
#                                     image[:,:,c]*
#                                     (1-alpha) + alpha*oil_color[c]*255,
#                                     image[:,:,c])
            
#             oil_image = image
    
#     if eyebag_result.mask is None :
#         eyebag_image = oil_image
        
#     else:
#         mask = eyebag_result.mask
                           
#         image = oil_image.copy()
#         image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                           
#         for c in range(3):
#             image[:,:,c] = np.where(mask==1,
#                                     image[:,:,c]*
#                                     (1-alpha) + alpha*eyebag_color[c]*255,
#                                     image[:,:,c])
            
#             eyebag_image = image

            
            
#     if eczema_result.mask is None :
#         eczema_image = eyebag_image
        
#     else:
#         mask = eczema_result.mask
        
#         image = eyebag_image.copy()
#         image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
#         for c in range(3):
#             image[:,:,c] = np.where(mask==1,
#                                     image[:,:,c]*
#                                     (1-alpha) + alpha*eczema_color[c]*255,
#                                     image[:,:,c])
            
#             eczema_image = image
    

    eczema_image = cv2.resize(eczema_image, (W, H), interpolation=cv2.INTER_LINEAR) # 원본 이미지 사이즈
                
    
    if len(acne_result.mask)==0 :
        acne_image = eczema_image

    else:
        for point in acne_result.mask:
            (h,w) = point
            acne_image = cv2.circle(eczema_image, (h,w), 20, (0, 0, 0), 4)
            
    if len(dent_result.mask)==0 :
        dent_image = acne_image

    else:
        for point in dent_result.mask:
            (h,w) = point
            dent_image = cv2.circle(acne_image, (h,w), 20, (49, 5, 39), 4)
            
    result_img = cv2.resize(dent_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    result_data = PILImage.fromarray(result_img)
        
    texture_image_data = skin_texture_plot(data_path=input_data_path, result_data=result_data, color=(255,105,180), thickness=4)
    texture_img = np.array(texture_image_data)
    overall_image = cv2.resize(texture_img, (W,H), interpolation=cv2.INTER_LINEAR)


    #-----------------------------------------------------------------------------------------------------------

    data = PILImage.fromarray(overall_image)
    s3_client.upload(folder=response.overall_image, file=data)
 
    #-----------------------------------------------------------------------------------------------------------

    return response
