# Face Skin AI (outsourcing project)       

2023/11/03 ~ 2024/04/03 (about 6 months)        

<table style="margin-left: auto; margin-right: auto;">
  <th>
    <p align='center'>Input</p>
  </th>
  <th>
    <p align='center'>Output</p>
  </th>
  <tr>
    <td>
      <p align='center'>
        <img src='https://silkycontents.s3.ap-northeast-2.amazonaws.com/media/face/pid_tjcowns@gmail.com/2024/13/56/KakaoTalk.Testinput.20240208_205531902_05.jpg' height='400'>
      <p>
    </td>
    <td>
      <p align='center'>
        <img src='https://github.com/WestChaeVI/Face_Skin_AI/test_output_overall.png' width='500'>
      <p>
    </td>
  </tr>
</table>   

--------------------------       




# 피부 분석 AI 모델 "Silky" Demo Version     

## Dataset : [Skin Problem Image Segmentation](https://universe.roboflow.com/hetvi-eww-zdjt2/skin-problem-image-segmentation/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)      
 
+ Acne : 142장       
+ Eye_bags : 80장     
+ Hyperpigment : 56장     
+ Blackhead : 26장      
+ Wrinkle : 59장      

> 현재 만들고자하는 모델을 학습시키기 위한 데이터셋을 찾기 위해 충분한 searching에도 불구하고, 적합한 데이터셋을 찾기는 어려웠다.      
> 우여곡절 끝에, 완벽하진 않지만 모델을 학습시킬 수는 있는 데이터셋을 선별함.      

## Data Analysis & Preprocessing      

+ 데이터가 어떤 형태로 저장되어 있고 라벨값은 어떻게 이루어져 있는지 분석 [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/EDA.ipynb)       
 > Labeled data는 배경을 포함한 6 class에 대해 일정하게 라벨링 작업이 되어 있지 않았고, 2개~6개 class로 다양하게 있었다.       
 > 그렇기 때문에, 배경을 포함한 6개의 class를 한 모델로 한번에 분류하는 것은 불가능했기에, 다음과 같은 전처리를 진행하였다.      

+ 분석을 토대로 난잡한 multi class label 에서 각각에 대한 binary label로 처리 [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/data_preprocessing.ipynb)       

 > Step 1      
   > Labeled Data 중에서 **0,k** 값만 가지고 있는 데이터를 가져와 k값을 1로 바꿔 binary mask로 바꾸기 **(k = 1, 2, 3, 4, 5)**     
 
 > Step 2    
   > binary mask와 짝을 이루고 있는 input data를 저장하기       

--------------------------------------------------------------------------------------------------------------------------------

## Train [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/acne_unet.ipynb)            

### Model        

+ Model은 Segmentation 초기 모델이자 기본 모델인 **U-Net**과 과도기의 모델인 **DeepLabV3+** 로 2개의 모델에 대해서 실험을 진행하였다.     

> U-Net 모델의 경우 Segmentation model의 시초인 FCN 모델의 인사이트를 차용하여 만든 모델로 여러 모델들에 비해 비교적 **가벼운 모델**에 속한다.     
>         
> 반대로 DeepLabV3+의 경우, 이름에서도 알 수 있듯이 Version을 세번이나 거듭하였고 Plus(+)까지 붙어 있는 것을 볼 수 있다. 과도기에 만들어진 모델로 **여러 좋은 모듈들을 혼합하여 만들어진 모델로 무거운 편**에 속한다.       
>        
> 현재 가지고 있는 dataset에 충분히 많지 않고, 정말 학습을 할 수만 있는 정도의 양이기 때문에, **무거운 모델에 학습할 경우 과적합이 일어나기 쉽다**.      
> 실험 결과, 예상대로 과적합이 일어났고, **성능은 두 모델 모두 낮은 성능을 보였으나, U-Net 모델이 더 높은 성능을 보여주었기에**, U-Net을 선정하게 되었다.     
  > 추후에, Dataset이 충분히 만들어진다면, DeepLabV3+ 모델을 쓰는 것이 성능면에서 더 좋다.     

## Data Loader      

1. 먼저, 데이터의 경로를 받아 원본 이미지와 마스크 이미지를 pair set으로 가져와 학습을 하기 위한 shape이나 dimension을 조정한다. [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/Codes/dataset.py)     
 > Why? | opencv 라이브러리를 통해 이미지를 읽게 되면 **numpy** 형태지만, 학습을 진행할 모델은 **torch** 형태이기 때문에 맞춰줘야 한다.       
2. 만들어진 dataset 모듈을 기반으로 데이터를 ipynb 환경에 불러오기 위한 data loader를 만든다. [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/Codes/data_loader.py)     
 > Augmentation       
 > Shuffle 유무       
 > Batch size       

+ Train, Valid, Test 은 6 : 2 : 2 로 split (dataset, data_loader 코드를 이용한 것이 아닌 사전에 코드로 split 해놓은 상태)      
+ Augmentation : Reszie (244 by 224), RandomRotation(180), RandomHorizontalFlip(0.5)      

### Set Up    

epoches : 400      
optimizer : Adam      
learning rate : 1e-4     
weight_decay : 1e-8      
criterion : DiceBCELoss [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/Codes/metrics.py)        


------------------------------------------------------------------------------------------------     

## Binary skin plot      

plot을 그리기 위한 [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/binary_skin_plot.py)      
Demo Version [Notebook](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/binary_demo.ipynb)

### Hyper-parameter      

+ input data path (테스트할 이미지의 경로)    
+ n_classes = 1 (최종 출력을 몇 개의 channel로 할 것인가)     
+ device (CUDA)     

### Pipline    

+ 각각의 class에 대해 binary task로 학습시킨 가중치를 각각 매겨 총 5개의 모델을 불러온다.     
+ 테스트 이미지는 Acne부터 Wrinkle까지 순차적으로 모델을 통과한다.    
+ 조건문을 통해, 예측된 mask 값들을 다시 0~5로 바꾼다.     
  > 예) 과다색소침착을 detect하는 모델을 통과한 mask라면, (0,1) -> (0,3)      
+ color       
  0 - Background     
  1 - Acne [빨강]      
  2 - Eye_bags [파랑]      
  3 - Hyperpigmentation(과다색소침착) [핑크]       
  4 - black_head [보라색]     
  5 - wrinkle [민트]      

+ 백분율(%) 계산 (소수점 둘째 자리까지)     

$$\frac{\text{mask의 해당 객체 픽셀 개수}}{\text{얼굴 영역의 픽셀 개수}} \ \text{x} \ 100 \  $$     


------------------------------------------------------------------------------------------------

# 추가 내용 - (2024.01.08)     

+ Acne - Segmentation $\rightarrow$ **Object Detection**       
+ Skin textrue - Mesh grid landmark     


## Acne Object Detection (YOLOv8) [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/acne_detection.ipynb)    

+ dataset : [roboflow](https://universe.roboflow.com/acne-severity/acne-detection-revisi)       
+ ultralytics libaray를 이용하여 yolov8 모델을 학습하고 새로운 이미지에 대해서 테스트까지 함.      

### YOLOv8 결과를 통해 bounding box -> circle      
<p align='center'><img src='https://github.com/eoncare-dev/silky-ai/assets/104747868/f4444600-5b24-483f-9cf2-89761a4faf0f'>     

+ bounding box의 x,y의 중점을 추출 -> 원의 중심 -> 원 그리기

<p align='center'><img src='https://github.com/eoncare-dev/silky-ai/assets/104747868/08d7e7d7-e2f2-44ba-9feb-656e7449796c'>     

## Skin texture - Mesh grid landmark [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/Codes/skin_texture.py)      

+ MediaPipe library를 사용해 mesh grid를 형성할 수 있었음.  
+ Steps     
 > 1. Mediapipe library를 사용하여 얼굴 부분에 mesh grid를 형성     
 > 2. grid의 spot과 index를 추출 (Index 값들은 general 한 값) [Code](https://github.com/eoncare-dev/silky-ai/blob/main/eonc/Codes/face_landmarks.py)     
 > 3. 전체 인덱스 값들 중, skin texture outline을 그려야할 포인트들을 찾음 (수작업)      
 > 4. 최종적으로 코 윗부분(upper nose)을 시작점으로 하여 한붓그리기를 수행    
 >      
 > color 수정 가능    
 > thickness 수정 가능

 ---

`data_loader.py`와 `dataset.py`는 학습 시 사용되는 코드들

`skin_plot.py`: 피부분석 함수를 포함한 `.py`파일. 해당 파일에서 `acne_spots.py`, `face_landmarks.py`, `skin_texture.py`를 호출하여 분석을 수행한다.

피부분석 진행 절차
1. `app.py`에서 API 요청 받기
2. `skin_plot.py`함수 호출
3. 5가지(acne, age_spot, texture, redness, wrinkle)의 피부분석 결과 이미지를 AWS S3로 저장
  - 저장 방식: pid_USEREMAIL > YEAR > MONTH > DAY > imageFIELD.UUID.png
4. 점수 계산과 이미지링크를 합산해서 `ResponseModel`로 응답.
