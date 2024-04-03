import cv2

def yolo_masking(resized_general_image, model_result, color, alpha=0.5):
    if model_result.mask is None :
        oil_image = resized_general_image
        
    else:
        mask = model_result.mask
        
        image = resized_general_image.copy()
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        for c in range(3):
            image[:,:,c] = np.where(mask==1,
                                    image[:,:,c]*
                                    (1-alpha) + alpha*oil_color[c]*255,
                                    image[:,:,c])
            
        return image