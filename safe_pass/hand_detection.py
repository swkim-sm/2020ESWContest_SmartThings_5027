# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from yolo import YOLO
#import serial

from win32api import GetSystemMetrics
system_width = GetSystemMetrics(0)
system_height = GetSystemMetrics(1)
def hand_detection(keyboard_roi, results, args):
    center = (0, 0)
    area = 0
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        area = w * h
        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        #cv2.rectangle(keyboard_roi, (x, y), (x + w, y + h), color, 2)
        center = (int((x + x + w) / 2), int((y + y + h) / 2))  # Detection 중심 계산
        cv2.circle(keyboard_roi, center, 3, (0, 0, 255), -1)  # Detection 중심 포인트 표시
        text = "%s (%s)" % (name, round(confidence, 2))
        #cv2.putText(keyboard_roi, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return center, area

# 손의 중심 포인트가 버튼 ROI에 포함되는지 확인하는 함수
def check_ROI(point, btn_roi):
    if point[0] + system_width*3/5 >= btn_roi[0] and point[0] + system_width*3/5 <= btn_roi[2]:
        X_CORRECT = True
    else:
        X_CORRECT = False

    if point[1] >= btn_roi[1] and point[1] <= btn_roi[3]:
        Y_CORRECT = True
    else:
        Y_CORRECT = False

    if X_CORRECT and Y_CORRECT:
        return True
    else:
        return False