# Library improt
import copy
import numpy as np

import torch

# Dataset improt
from train.data_loader import Binary_Data_loader

# Model import
from model.Unet.unet_models import UNet
from model.Deeplab.Deeplabv3 import deeplabv3_resnet101

# Metrics import
from train.metrics import dice_score ,iou ,dice_pytorch_eval, iou_pytorch_eval, DiceBCELoss
from train.metrics import DiceLossMulticlass

# cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
      torch.cuda.manual_seed_all(42)
        
        
epoches = 400
model = UNet(n_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)
criterion = DiceBCELoss()


def train(objective, batch_size=8):
    
    train_loss_list, train_dice_list, train_iou_list = [], [], []
    valid_loss_list, valid_dice_list, valid_iou_list = [], [], []
    state={}
    model_name = 'Unet' # deeplabv3로 전환
    n_classes = 1
    
    # 데이터셋 로드
    train_loader, valid_loader, test_loader = Binary_Data_loader(f'/home/sh/lab/eonc/data/multi/skin/{objective}', batch_size)  # 데이터 경로 지정 확인, 8 : batch_size

    for epoch in range(1, epoches + 1) :
        train_loss, train_dice, train_iou, train_step = 0, 0, 0, 0

        # Train
        model.train()
        for images, mask in train_loader:
            images = images.to(device)
            mask = mask.to(device)

            # 모델에 이미지 입력
            pred = model(images)  # images.shape : [8, 3, 224, 224]
                                  # pred.shape : [8, 1, 224, 224] 

            # loss function 학습
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if n_classes==1:
                train_iou += iou_pytorch_eval(pred, mask).item()
                train_dice += dice_pytorch_eval(pred, mask).item()

            else:
                train_iou += iou(pred, mask).item()
                train_dice += dice_score(pred, mask).item()
            train_step += 1

        # Validate
        val_loss, val_dice, val_iou, val_step = 0, 0, 0, 0

        model.eval()
        with torch.no_grad():

            for image, mask in valid_loader:
                image = image.to(device)
                mask = mask.to(device)
                
                # 모델에 입력
                mask_pred = model(image)

                loss = criterion(mask_pred, mask)
                val_loss += loss.item()
                if n_classes==1:
                    val_iou += iou_pytorch_eval(mask_pred, mask).item()
                    val_dice += dice_pytorch_eval(mask_pred, mask).item()

                else:
                    val_iou += iou(mask_pred, mask).item()
                    val_dice += dice_score(mask_pred, mask).item()
                val_step += 1     

    # ------------------------------------------------------------------------------------------------        

        train_dice = round(train_dice / train_step, 3)      
        train_iou = round(train_iou / train_step, 3)
        train_loss = round(train_loss / train_step, 3)

        val_dice = round(val_dice / val_step, 3)
        val_iou = round(val_iou / val_step, 3)
        val_loss = round(val_loss / val_step, 3)

        print(f'Epoch: {epoch}/{epoches}, Train Dice: {train_dice}, Train Iou: {train_iou}, Train Loss: {train_loss}')
        print(f'Epoch: {epoch}/{epoches}, Valid Dice: {val_dice}, Valid Iou: {val_iou}, Valid Loss: {val_loss}\n')

        # save list
        train_dice_list.append(train_dice)
        train_iou_list.append(train_iou)
        train_loss_list.append(train_loss)

        valid_dice_list.append(val_dice)
        valid_iou_list.append(val_iou)
        valid_loss_list.append(val_loss)

        # overpitting 막는 if문
        if np.max(valid_dice_list) <= val_dice:
            state['epoch'] = epoch
            state['net'] = copy.deepcopy(model.state_dict())

            state['train_dice'] = train_dice
            state['train_iou'] = train_iou
            state['train_loss'] = train_loss

            state['val_dice'] = val_dice
            state['val_iou'] = val_iou
            state['val_loss'] = val_loss

    torch.save(state, '/home/sh/lab/eonc/check_points/{}_{}_{}_{}.pth'.format(objective, model_name, state['epoch'], state['val_dice']))  # 저장할 경로 지정 확인
    print('total best epoch : {} / dice_score : {} / Iou : {} / Loss : {}'.format(state['epoch'], state['val_dice'],state['val_iou'],state['val_loss']))