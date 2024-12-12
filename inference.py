from ultralytics import YOLO
from PIL import Image



model = YOLO("D:/Yolov8/Save_final/weights/last.pt") 

source = Image.open("D:/Yolov8/TestImage/test1.png")

result = model.predict(source, save=True, conf=0.7)
