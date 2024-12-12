## YOLOv8 모델

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO
model = YOLO("yolov8x.yaml")  # build a new model from scratch

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.train(data="D:/Yolov8/ultralytics/Yolov8_UI-similarity_ver4-1/data.yaml", 
            epochs=300, 
            device=0, 
            batch=8, 
            imgsz=640, 
            workers=0, 
            cache=False, 
            name="D:/Yolov8/Save_final", 
            exist_ok=True,
            plots=True  # 그래프 저장 활성화
            )  # train the model
metrics = model.val()  # evaluate model performance on the validation set

metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category

metrics.results_dict

import json

# metrics.results_dict를 JSON 파일로 저장
with open('metrics_results.json', 'w') as f:
    json.dump(metrics.results_dict, f)


