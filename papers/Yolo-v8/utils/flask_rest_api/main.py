import cv2
from ultralytics import YOLO

#load a model
model = YOLO("/nvme0/huangzeliang/xianyinggunYoloV8/xianyinggunV8/runs/segment/8m/weights/best.pt")

results = model("/nvme0/huangzeliang/xianyinggunYoloV8/datasets/testimg")

print("h",results)