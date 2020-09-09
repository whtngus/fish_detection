# fish_detection
fish_image_detection

# 1. 데이터 가공 준비

- 준비되어 있는 데이터는 labelimg 프로그램의 yolo포맷으로 만든 데이터 입니다.  <br>

> 따라서 다른 포맷의 데이터를 사용한다면 변형하여 작업 부탁드립니다. <br>

### 데이터 라벨링 <br>

- 처음 물고기 데이터를 태깅할 때 태깅 속도를 빠르게 하기위해 object detection pretrained model을 사용하여 1차 가공을 사용한 후 디텍션 합니다. <br>

##### 사용한 코드 <br>

> https://github.com/kwea123/fish_detection <br>

##### 사용 방법 <br>

코드 위치 : /src/model/pre_detection/main.py <br>

- 실행 예 <br>
python3 ./src/model/pre_detection/main.py -data "학습 데이터 경로" -model "학습된 모델 경로" -show_image "태깅중인 이미지를 볼지 여부 default False" <br>

- 모델 및 데이터  <br>

> - 데이터 <br>
> 대상 디렉토리에 아래 예시와 같이 물고기명 디렉토리로 있어야 하고  그안에 사진들이 들어있어야 한다. <br>
<img src="./picture/data_example_1.JPG" width="500px" height="400px"></img>  <br>
> 전부 학습 후 아래 예시와 같이 labelimg 툴에서 사용하는 yolo형식의 결과물이 생성된다 <br>
<img src="./picture/data_example_2.JPG" width="500px" height="400px"></img>  <br>

> - 모델 <br>
> 사용한 모델 :  fish_inception_v2_graph/frozen_inference_graph.pb <br>
> 모델 다운로드 방법은 https://github.com/kwea123/fish_detection 참조 <br>


# 2. 가공된 데이터로 학습 준비 

- labelimg 툴로 yolo방식으로 라벨링 된 데이터를 retinanet으로 학습하기 위한 포맷으로 변경

##### 사용 방법 <br>

코드 위치 : /src/data_manage/change_data_form.py <br>

*코드 상단의 base_path 변수에 파일 위치로 변경*<br>

- 실행 예 <br>
python3 ./src/data_manage/change_data_form.py <br>

> - 데이터 <br>
> 대상 디렉토리에 아래 예시와 같이 물고기명 디렉토리로 있어야 하고  그안에 사진들이 들어있어야 한다. <br>
<img src="./picture/data_example_1.JPG" width="500px" height="400px"></img>  <br>
> 전부 처리 후 retinanet을 학습하기 위한 포맷인 data.csv 파일이 생성된다 <br>
*가공이 안되거나 문제가 있는 데이터의 경우 아래와 같이 프린트 출력딤*  <br>



# 3. 학습 및 평가 방법 

##### 사용한 코드 <br>

git url :  https://github.com/fizyr/keras-retinanet <br>
*자세한 학습 방법과 환경 설정은 위의 github를 참조 바랍니다<br>

-  실행 예시 <br>
train.py   --gpu=0 --epochs=50 --steps=459 --batch-size=3 --workers=0 csv ./resources/images/obj/data.csv ./resources/images/obj/class_name.csv <br> 
<br>
1. 학습 파일 <br>
src\model\retinanet\keras_retinanet\bin\train.py <br>
2. 파라미터 예시 <br>

--gpu=0  --steps=1414 --batch-size=2 --workers=0 csv ./resources/images/obj/data.csv ./resources/images/obj/class_name.csv --val-annotations ./resources/images/eval_obj/data.csv <br>

> steps size * batch-size 는 데이터의 크기로 지정 <br>

3. 파일 예시 <br>

```
    - data.csv
bass/1589464782993-11.jpg,468,1,1032,1049,bass
bass/1589464782993-12.jpg,429,1,815,1076,bass
bass/1589464782993-13.jpg,381,0,867,1295,bass
...
...

    - class_name.csv
black_porgy,0
rockfish,1
tripletail,2
bass,3
mackerel,4
rock_bream,5
black_rock_fish,6
gray_mullet,7
threeline_grunt,8
girella_punctata,9
flatfish,10
spotty_belly_greenling,11
flounder,12
spanish_mackerel,13
scorpion_fish,14
croaker,15
bass_freshwater_fish,16
bluegill_freshwater_fish,17
convict_grouper,18
snake_head_freshwater_fish,19
korean_bullhead_freshwater_fish,20
carp_freshwater_fish,21
mandarin_fish_freshwater_fish,22
...
... 
```


# 4. 학습한 모델 사용하기

1. retinanet 모델 변경하기 <br>
> 파일 경로 : src\modules\retinanet\keras_retinanet/bin/convert_model.py 
> 사용 예시 <br>
> convert_model.py  ./snapshots/resnet50_csv_25.h5 ./snapshots/resnet50_csv_25_infer.h5

2. 파라미터 <br>
"대상 모델" "output model" <br>
