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

# Skin detect and Noise reduction
def detectSkin(original):
    # Color binarization
    ycbcr = cv2.cvtColor(src=original, code=cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 85, 135])
    upper = np.array([255, 135, 180])
    mask = cv2.inRange(ycbcr, lower, upper)

    # Noise reduction - Median Filter
    mask = cv2.medianBlur(src=mask, ksize=7)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(7, 7))
    mask = cv2.morphologyEx(src=mask, op=cv2.MORPH_CLOSE, kernel=element)
    mask = cv2.morphologyEx(src=mask, op=cv2.MORPH_OPEN, kernel=element)
    return mask


# 손가락 개수와 손가락 사이 각을 이용한 제스쳐 인식
def getHandGesture(points):
    fingerCount = len(points) - 1
    result = ""
    if fingerCount == 0:
        result = "Rock"  # 주먹
    elif fingerCount == 1:
        result = "One"  # 손가락 하나
    elif fingerCount == 2:
        result = "Two"  # 손가락 둘
    elif fingerCount == 3:
        result = "Three"
    elif fingerCount == 4:
        result = "Four"
    elif fingerCount == 5:
        result = "Five"
    return result, fingerCount


# Find out and draw Hand Gesture
def drawHandGesture(frame, mask, keyboard_x, keyboard_y):
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    color = (0, 255, 255)
    thickness = 1

    mask = cv2.medianBlur(src=mask, ksize=7)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(7, 7))
    mask = cv2.morphologyEx(src=mask, op=cv2.MORPH_CLOSE, kernel=element)
    mask = cv2.morphologyEx(src=mask, op=cv2.MORPH_OPEN, kernel=element)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_radius = 0
    motion_num = -1
    for cnt in contours:
        # 객체 외곽선 그리기
        cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 3)  # blue

        # 손바닥 중심과 반지름 찾기
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        scale = 0.7
        center = (int(x), int(y))
        radius = int(radius)
        if radius < 100:
            continue
        if max_radius < radius:
            max_radius = radius
        # cv2.circle(frame, center, 2, (0, 255, 0), -1)
        # cv2.circle(frame, center, int(radius*scale), (0, 255, 0), 2)

        # 손가락 개수를 세기 위한 원 그리기
        cImg = np.zeros(mask.shape, np.uint8)
        cv2.circle(cImg, center, int(radius*scale), 255)

        # 원의 외곽선을 저장할 벡터
        circleContours, hierarchy = cv2.findContours(cImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 원의 외곽선을 따라 돌며 mask의 값이 0에서 1로 바뀌는 지점 확인
        points = []
        for i in reversed(range(1, len(circleContours[0]))):
            p1 = (circleContours[0][i][0][0], circleContours[0][i][0][1])
            p2 = (circleContours[0][i-1][0][0], circleContours[0][i-1][0][1])
            if mask[p1[1], p1[0]] == 0 and mask[p2[1], p2[0]] > 1:
                points.append(p1)

        # 손가락 마디(선) 그리기
        for i in range(len(points)):
            cv2.line(frame, center, points[i], (0, 255, 0), 3)

        # 제스처 출력
        text, motion_num = getHandGesture(points)
        cv2.putText(frame, text, center, font, fontScale, color, thickness)

    return frame, motion_num, max_radius

def make_button(x, y, width, height, text):
    th = 3 # thickness
    cv2.rectangle(frame, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(frame, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)

    return frame[x + th : x + width - th, y + th : y + height - th]

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
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

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

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1100)
    frame = cv2.flip(frame, 1)
    # mask 미착용 또는 확인 단계 
    if not wearing_mask:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)


            if label == "Mask" and not mask_flag: # 마스크 착용 시간 재기
                mask_flag = True
                start = time.time()
            elif label == "Mask": # 3초간 마스크 확인
                during = time.time()-start
                cv2.putText(frame, "checking : " + str(during), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                if during > 3:
                    wearing_mask = True
                    mask_flag = False
            else: # 마스크 미착용 시 경고 문구
                cv2.putText(frame, "Please wear a MASK!!!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                mask_flag = False

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # mask 착용 확인 후 virtual keyboard 조작 단계
    else:
        # keyboard 생성
        keyboard_x = 600
        keyboard_y = 200
        keyboard_roi = frame[0:1000, 500:1100]
        hand_roi = detectSkin(keyboard_roi)
        frame, hand_result, hand_radius = drawHandGesture(frame, hand_roi, keyboard_x, keyboard_y)
        display = cv2.rectangle(frame, (keyboard_x+3, keyboard_y+3), (keyboard_x+400-3, keyboard_y-150-3), (255, 0, 0), 3)
        btn_A = make_button(keyboard_x, keyboard_y, 200, 200, "A")
        btn_B = make_button(keyboard_x+200, keyboard_y, 200, 200, "B")
        btn_C = make_button(keyboard_x, keyboard_y+200, 200, 200, "C")
        btn_D = make_button(keyboard_x+200, keyboard_y+200, 200, 200, "D")
        btn_end = make_button(keyboard_x, keyboard_y+400, 400, 150, "End")
        
        # hand gesture 로 keyboard 클릭
        #cv2.rectangle(frame, (keyboard_x, keyboard_y), (keyboard_x+400, keyboard_y+600), (255, 255, 0), 2)
        
        


## To do
#### 버튼마다 버튼안에 손바닥 들어가면 색깔 바뀌게
#### 버튼에서 손가락이 0이면 클릭하기
        



    # show the output frame and break the loop by button 'q'
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

'''
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

    if not wearing_mask:

        # detect faces in the frame and determine if they are wearing face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    else:
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
'''
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
