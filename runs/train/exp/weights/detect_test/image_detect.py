# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 01:58:59 2025

@author: kimke
"""

# predict_with_custom_model.py

import os
import torch
from ultralytics import YOLO

import platform
import pathlib

# Check the operating system and set the appropriate path type
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# 1. 학습된 모델 불러오기
# 학습 완료 후 생성된 'best.pt' 파일을 로드합니다.
# 이 파일은 학습 과정에서 가장 좋은 성능을 보인 모델의 가중치입니다.
# 경로는 'runs/detect/cat_dog_yolov8n/weights/best.pt'와 유사합니다.

# 가중치 파일 경로 

TRAINED_MODEL_PATH = '../best.pt'


model = torch.hub.load('ultralytics/yolov5', 'custom', TRAINED_MODEL_PATH, force_reload=True)

# 모델을 GPU로 이동 (선택 사항)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 2. 예측할 이미지 경로 지정
# 예시: 'data/test/my_new_image.jpg'
IMAGE_PATH = 'potholes9_png.rf.fe92e8fa27cd17ea446b44207726a8da.jpg'

results = model(IMAGE_PATH)


results.show()
