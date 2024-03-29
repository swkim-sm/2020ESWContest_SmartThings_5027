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

# 버튼 rectangle 생성하는 함수 - clicked 여부에 따라 분류
def make_button(frame, x, y, width, height, color, text, clicked):
    th = 3  # thickness
    if clicked:
        cv2.rectangle(frame, (x + th, y + th), (x + width - th, y + height - th), (0, 0, 255), -1)
    else:
        cv2.rectangle(frame, (x + th, y + th), (x + width - th, y + height - th), color, th)

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 4
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(frame, text, (text_x, text_y), font_letter, font_scale, color, font_th)

    return x, y, x + width, y + height, text