import cv2
import torch
import numpy as np
from PIL import Image as PILImage
from settings.s3 import MyS3Client
from code.skin_texture import skin_texture_plot, face_recognize
from code.utils import get_filename
from code.image_processor import image_preprocess, image_masking

from model.common.acne import acne_detect_analysis
from model.common.age_spot import age_spot_analysis
from model.common.redness import redness_analysis
from model.common.texture import texture_analysis
from model.common.wrinkle import wrinkle_analysis
from model.common.dent import dent_analysis
from model.common.oil import oil_analysis
from model.common.eyebag import eyebag_analysis
from model.common.eczema import eczema_analysis

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
    
#------------------------------------------------------------------------------------------------------------------


def overall(email, input_data_path):
    
    '''
    <순서>
    age_spot - redness - texture - wrinkle
    oil - eyebag - eczema - acne - dent - skin_texture_plot
    
    segmentation 영역들 다 그리고 나서 동그라미 그린 후 skin_texture_plot
    '''
    
    s3_client = MyS3Client()
    general_img, eval_image = image_preprocess(input_data_path, device)
    overall_image_path = get_filename(email, field='overall')
    
    H,W,C = general_img.shape   # (H, W ,3)
    b,c,h,w = eval_image.shape  # (1, 3, 224, 224)
    
    acne_result = acne_detect_analysis(email, input_data_path)  # point_list
    age_spot_result = age_spot_analysis(email, input_data_path) # 224,224,3
    redness_result = redness_analysis(email, input_data_path)   # 224,224,3
    texture_result = texture_analysis(email, input_data_path)   # 224,224,3
    wrinkle_result = wrinkle_analysis(email, input_data_path)   # 224,224,3
    dent_result = dent_analysis(email, input_data_path)         # point_list
    oil_result = oil_analysis(email, input_data_path)           # (640, 608)
    eyebag_result = eyebag_analysis(email, input_data_path)     # (640, 608)
    eczema_result = eczema_analysis(email, input_data_path)     # (640, 608)
    
    
    resized_general_img = cv2.resize(general_img, (w,h), interpolation=cv2.INTER_LINEAR)
    
    resized_general_img += age_spot_result.mask
    resized_general_img += redness_result.mask
    resized_general_img += texture_result.mask
    resized_general_img += wrinkle_result.mask
    
    resized_general_img = cv2.resize(resized_general_img, (608, 640), interpolation=cv2.INTER_LINEAR)
    
    alpha = 0.5
    oil_color = (135/255,206/255,235/255)
    eyebag_color = (128/255,128/255,128/255)
    eczema_color = (255/255,10/255,100/255)
    
    if oil_result.mask is None :
        oil_image = resized_general_img
        
    else:
        mask = oil_result.mask
        
        image = resized_general_img.copy()
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        for c in range(3):
            image[:,:,c] = np.where(mask==1,
                                    image[:,:,c]*
                                    (1-alpha) + alpha*oil_color[c]*255,
                                    image[:,:,c])
            
            oil_image = image
            
            
    if eyebag_result.mask is None :
        eyebag_image = oil_image
        
    else:
        mask = eyebag_result.mask
                           
        image = oil_image.copy()
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                           
        for c in range(3):
            image[:,:,c] = np.where(mask==1,
                                    image[:,:,c]*
                                    (1-alpha) + alpha*eyebag_color[c]*255,
                                    image[:,:,c])
            
            eyebag_image = image

            
            
    if eczema_result.mask is None :
        eczema_image = eyebag_image
        
    else:
        mask = eczema_result.mask
        
        image = eyebag_image.copy()
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        for c in range(3):
            image[:,:,c] = np.where(mask==1,
                                    image[:,:,c]*
                                    (1-alpha) + alpha*eczema_color[c]*255,
                                    image[:,:,c])
            
            eczema_image = image
    

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
            
    
    data = PILImage.fromarray(overall_image[:,:,::-1])
    s3_client.upload(folder=overall_image_path, file=data) 
    
    
    return overall_image_path