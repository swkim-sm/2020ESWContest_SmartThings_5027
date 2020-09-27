# USAGE
# python detect_mask_video.py

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
import serial

from win32api import GetSystemMetrics


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


# 버튼 rectangle 생성하는 함수 - clicked 여부에 따라 분류
def make_button(x, y, width, height, color, text, clicked):
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


def update_status(current_status, clicked_btn):
    # on 버튼
    if (clicked_btn == 3):
        current_status[clicked_btn] = 1

    # off 버튼
    else:
        current_status[clicked_btn] = 0


def hand_detection(keyboard_roi, results):
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


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument('-n', '--network', default="tiny", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-hc', '--handconfidence', default=0.2, help='Confidence for yolo')
args = vars(ap.parse_args())
# args = ap.parse_args()

if args["network"] == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args["network"] == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
yolo.size = int(args["size"])
yolo.confidence = float(args["handconfidence"])

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
wearing_mask = False
motion_end = False
mask_flag = False
prev_btn = -1

# logo image
logo_img = cv2.imread("logo.png")
# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = logo_img.shape
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
ret, logo_mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(logo_mask)

# led, fan, belt button off : 0 / on : 1
current_status = [0, 0, 0]
# arduino board
#ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
# ser = serial.Serial('COM4', 115200, timeout=1)
# ser.open()

# flag for clicking button
before_area = 0
click_check_ratio = 1.4

system_width = GetSystemMetrics(0)
system_height = GetSystemMetrics(1)
keyboard_x = int(system_width*3/5)
keyboard_y = 0
print("Width =", system_width)
print("Height =", system_height)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=int(system_width), height=int(system_height))
    frame = cv2.flip(frame, 1)
    # mask 미착용 또는 확인 단계
    if not wearing_mask:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            if label == "Mask" and not mask_flag:  # 마스크 착용 시간 재기
                mask_flag = True
                start = time.time()
            elif label == "Mask":  # 2초간 마스크 확인
                during = time.time() - start
                cv2.putText(frame, "checking : " + str(during), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if during > 2:
                    wearing_mask = True
                    mask_flag = False
            else:  # 마스크 미착용 시 경고 문구
                cv2.putText(frame, "Please wear a MASK!!!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                mask_flag = False

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # mask 착용 확인 후 virtual keyboard 조작 단계
    else:
        # keyboard 생성
        keyboard_roi = frame[0:int(system_height), keyboard_x:int(system_width)]

        click_flag = False
        width, height, inference_time, results = yolo.inference(keyboard_roi)
        center, area = hand_detection(keyboard_roi, results)
        btn_list = []

        # cv2.putText(frame, "Please select a device to operate", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
        color_white = (255, 255, 255)
        color_gray = (200, 200, 200, 50)

        if prev_btn == 0:
            btn_LED = make_button(keyboard_x + 50, keyboard_y + 200, 200, 150, color_white, "LED", True)
            btn_FAN = make_button(keyboard_x + 50, keyboard_y + 400, 200, 150, color_gray, "FAN", False)
            btn_BELT = make_button(keyboard_x + 50, keyboard_y + 600, 200, 150, color_gray, "BELT", False)
            btn_ON = make_button(keyboard_x + 300, keyboard_y + 200, 200, 150,  color_white,"ON", False)
            btn_OFF = make_button(keyboard_x + 300, keyboard_y + 400, 200, 150,  color_white,"OFF", False)
            btn_END = make_button(keyboard_x + 300, keyboard_y + 600, 200, 150,  color_white,"END", False)
        elif prev_btn == 1:
            btn_LED = make_button(keyboard_x + 50, keyboard_y + 200, 200, 150, color_gray,"LED", False)
            btn_FAN = make_button(keyboard_x + 50, keyboard_y + 400, 200, 150, color_white,"FAN", True)
            btn_BELT = make_button(keyboard_x + 50, keyboard_y + 600, 200, 150, color_gray,"BELT", False)
            btn_ON = make_button(keyboard_x + 300, keyboard_y + 200, 200, 150,  color_white,"ON", False)
            btn_OFF = make_button(keyboard_x + 300, keyboard_y + 400, 200, 150,  color_white,"OFF", False)
            btn_END = make_button(keyboard_x + 300, keyboard_y + 600, 200, 150,  color_white,"END", False)
        elif prev_btn == 2:
            btn_LED = make_button(keyboard_x + 50, keyboard_y + 200, 200, 150, color_gray,"LED", False)
            btn_FAN = make_button(keyboard_x + 50, keyboard_y + 400, 200, 150, color_gray,"FAN", False)
            btn_BELT = make_button(keyboard_x + 50, keyboard_y + 600, 200, 150, color_white,"BELT", True)
            btn_ON = make_button(keyboard_x + 300, keyboard_y + 200, 200, 150,  color_white, "ON", False)
            btn_OFF = make_button(keyboard_x + 300, keyboard_y + 400, 200, 150,  color_white, "OFF", False)
            btn_END = make_button(keyboard_x + 300, keyboard_y + 600, 200, 150,  color_white, "END", False)
        else:
            btn_LED = make_button(keyboard_x + 50, keyboard_y + 200, 200, 150, color_white, "LED", False)
            btn_FAN = make_button(keyboard_x + 50, keyboard_y + 400, 200, 150, color_white, "FAN", False)
            btn_BELT = make_button(keyboard_x + 50, keyboard_y + 600, 200, 150, color_white, "BELT", False)
            btn_ON = make_button(keyboard_x + 300, keyboard_y + 200, 200, 150, color_gray, "ON", False)
            btn_OFF = make_button(keyboard_x + 300, keyboard_y + 400, 200, 150, color_gray, "OFF", False)
            btn_END = make_button(keyboard_x + 300, keyboard_y + 600, 200, 150, color_white, "END", False)

        btn_list.append(btn_LED)
        btn_list.append(btn_FAN)
        btn_list.append(btn_BELT)
        btn_list.append(btn_ON)
        btn_list.append(btn_OFF)
        btn_list.append(btn_END)


        # 손 면적 여부를 통한 클릭 여부 확인
        if (area >= int(before_area * click_check_ratio)):  # 클릭 후 손 펴는 동작으로 면적이 커졌을 시
            click_flag = True

        if click_flag:  # 클릭 발생 시
            for btn in range(0, len(btn_list)):
                if check_ROI(center, btn_list[btn]):
                    make_button(btn_list[btn][0], btn_list[btn][1], btn_list[btn][2] - btn_list[btn][0],
                                btn_list[btn][3] - btn_list[btn][1], color_white, btn_list[btn][4], True)
                    if (btn in (0, 1, 2)):
                        prev_btn = btn
                    elif btn == 5:
                        wearing_mask = False
                    else:
                        # on 버튼
                        if btn == 3:
                            if current_status[prev_btn] == 0:
                                current_status[prev_btn] = 1
                                print(prev_btn * 2 + 1)
                                text = str(prev_btn * 2 + 1)
                                text = bytes(text, 'utf-8')
                                # ser.write(text)

                        # off 버튼
                        else:
                            if current_status[prev_btn] == 1:
                                current_status[prev_btn] = 0
                                print(prev_btn * 2 + 2)
                                text = str(prev_btn * 2 + 2)
                                text = bytes(text, 'utf-8')
                                # ser.write(text)

        before_area = area
        click_flag = False

    # logo
    roi = frame[50:rows + 50, keyboard_x-10:cols + keyboard_x-10]
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(logo_img, logo_img, mask=logo_mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    frame[50:rows + 50, keyboard_x-10:cols + keyboard_x-10] = dst

    # show the output frame and break the loop by button 'q'
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        wearing_mask = False
        motion_end = False
        mask_flag = False
        current_status = [0, 0, 0]
        prev_btn = -1

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()