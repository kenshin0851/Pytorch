# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 01:16:58 2025

@author: kimke
"""

import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 위에서 Trained된 모델 불러오기
model = YOLO('best.pt')

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
    results = model.predict(source = img)
    plots = results[0].plot()
    cv2.imshow("PC_Webcam",plots)
    # esc 누르면 종료하기
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()