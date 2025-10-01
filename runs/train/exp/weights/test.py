# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 01:16:58 2025

@author: kimke
"""

import os
import cv2
import torch
from ultralytics import YOLO



import platform
import pathlib

# Check the operating system and set the appropriate path type
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath



# 위에서 Trained된 모델 불러오기
TRAINED_MODEL_PATH = 'best.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', TRAINED_MODEL_PATH, force_reload=True)
# 카메라 켜기
cap = cv2.VideoCapture(0)
# 웹캠 오류 처리
if not cap.isOpened():
    print("camera open failed")
    exit()

# inference 하고 싶은 이미지 불러오기
while True:
    # 카메라 읽기
    status, img = cap.read()
    if not status:
        print("Can't read camera")
        break
    # 카메라 창 띄우기
    results = model(img[:, :, ::-1])
    rendered_frame = results.render()[0]
    
    cv2.imshow('Detections', rendered_frame[:, :, ::-1])
    
    # esc 누르면 종료하기
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()