import glob
import cv2
import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F
from torchvision import transforms, datasets


def deeplabv3_resnet101(outchannels=6):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if device == 'cuda':
          torch.cuda.manual_seed_all(42)

    model = torchvision.models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
    
    # fine tuning
    model.classifier[4] = nn.Conv2d(256, 6, kernel_size=(1,1), stride=(1,1)) # 다뤄야할 클래스 개수 6개
    
    return model