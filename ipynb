#Скачаем наш датасет
!wget -c http://ai.stanford.edu/~jkrause/car196/car_ims.tgz -O - | tar -xz
!wget -c http://ai.stanford.edu/~jkrause/car196/cars_annos.mat

#Сконвертируем датасет в формат YOLO
import scipy.io as io
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

fname ='/content/cars_annos.mat'
data  = io.loadmat(fname)
%cd /content/

path_train = '/content/train/'
path_test = '/content/test/'
path_val = '/content/val/'

labels_path_train = path_train + 'labels/'
labels_path_test = path_test + 'labels/'
labels_path_val = path_val + 'labels/'
img_path_train = path_train + 'images/'
img_path_test = path_test + 'images/'
img_path_val = path_val + 'images/'

for path in [labels_path_train, labels_path_test, 
             img_path_train, img_path_test,
             labels_path_val, img_path_val]:
    os.makedirs(path, exist_ok=True)

def parse_data(data):
    '''Распарсим данные и разложим их в 3 кучки. Train, test и val'''
    for num, i in tqdm(enumerate(range(len(data['annotations'][0])))):
        sample = data['annotations'][0][i]

        img_path = sample[0][0]
        bbox_x1 = sample[1][0][0]
        bbox_y1 = sample[2][0][0]
        bbox_x2 = sample[3][0][0]
        bbox_y2 = sample[4][0][0]
        class_label = sample[5][0][0]-1
        test = sample[6][0][0]

        image = Image.open(img_path)
        width, height = image.size

        label_name = img_path.split('/')[-1].split('.')[0]

        x1 = bbox_x1/width
        x2 = bbox_x2/width
        y1 = bbox_y1/height
        y2 = bbox_y2/height

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        labels_path = labels_path_train
        img_path_final = img_path_train
        if num in np.random.randint(size=(200), low=0, high=len(data['annotations'][0])):
            labels_path = labels_path_val
            img_path_final = img_path_val
        if num in np.random.randint(size=(1000), low=0, high=len(data['annotations'][0])):
            labels_path = labels_path_test
            img_path_final = img_path_test
        with open(labels_path + label_name + '.txt', mode="w") as label_file:
            label_file.write(
                f"{class_label} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
            )
            os.rename(img_path, img_path_final + label_name + '.jpg')
        
parse_data(data)

## Создадим файл с настройками для нашей YOLO

import yaml
%cd /content/
names = np.array([x[0] for x in data['class_names'][0]]).tolist()

setting = dict(
    train = path_train,
    val = path_val,
    test = path_test,
    nc = 196,
    names = names
)

with open('/content/settings.yaml', 'w') as outfile:
    yaml.dump(setting, outfile, default_flow_style=False, )
    
## Скачаем YOLOv5
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

# !pip install wandb # Для мониторинга обучения
## Затюним нашу предобученную YOLOv5. Будем делать resize изображений, что бы быстрее училась и все влезло в память

%cd /content/yolov5/
!python train.py --batch 48 --weights yolov5l.pt --data /content/settings.yaml --epochs 50 --cache --img 256 

#Посчитаем точность на тестовой выборке
import yaml
%cd /content/
names = np.array([x[0] for x in data['class_names'][0]]).tolist()

setting = dict(
    train = path_train,
    val = path_test,
    nc = 196,
    names = names
)

with open('/content/settings_test.yaml', 'w') as outfile:
    yaml.dump(setting, outfile, default_flow_style=False, )
    
trained_weights = '/content/yolov5/runs/train/exp3/weights/best.pt'

%cd /content/yolov5/
!python test.py --batch 48 --data /content/settings_test.yaml --weights $trained_weights --conf 0.25 --img 256

#Будем предсказывать наши машинки

url = 'https://www.supercars.net/blog/wp-content/uploads/2016/04/2012_AstonMartin_V8VantageRoadster1.jpg'
!wget $url 

!python detect.py --source /content/yolov5/2012_AstonMartin_V8VantageRoadster1.jpg --weights $trained_weights --img 256 --max-det 1

url2 = 'https://i.pinimg.com/originals/b6/9c/b2/b69cb232b50b1226a9245e19b0e4d455.jpg'
!wget $url2

!python detect.py --source /content/yolov5/b69cb232b50b1226a9245e19b0e4d455.jpg --weights $trained_weights --img 256 --max-det 1

!python detect.py --source /content/val/images --weights /content/yolov5/runs/train/exp3/weights/best.pt --img 256 --max-det 1

url3 = "https://www.bmw.ru/content/dam/bmw/common/all-models/3-series/sedan/2018/navigation/bmw-3-series-modellfinder.png/_jcr_content/renditions/cq5dam.resized.img.585.low.time1579021049723.png"
!wget $url3

!python detect.py --source /content/yolov5/cq5dam.resized.img.585.low.time1579021049723.png --weights /content/yolov5/runs/train/exp3/weights/best.pt --img 256 --max-det 1