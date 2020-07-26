# -*- coding: utf-8 -*-
import cv2 as cv
import math
import numpy as np

# Skin detect and Noise reduction
def detectSkin(original):
	# Color binarization
	ycbcr = cv.cvtColor(src=original, code=cv.COLOR_RGB2YCrCb)
	lower = np.array([0, 85, 135])
	upper = np.array([255, 135, 180])
	mask = cv.inRange(ycbcr, lower, upper)

	# Noise reduction - Median Filter
	mask = cv.medianBlur(src=mask, ksize=7)
	element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(7, 7))
	mask = cv.morphologyEx(src=mask, op=cv.MORPH_CLOSE, kernel=element)
	mask = cv.morphologyEx(src=mask, op=cv.MORPH_OPEN, kernel=element)
	return mask


# 점 a1,a2를 잇는 직선과 점 b1,b2를 잇는 직선 사이의 각 반환
def getAngle(a1, a2, b1, b2):
	dx1 = a1[0] - a2[0]
	dy1 = a1[1] - a2[1]
	dx2 = b1[0] - b2[0]
	dy2 = b1[1] - b2[1]

	rad = math.atan2(dy1*dx2 - dx1 * dy2, dx1*dx2 + dy1 * dy2)
	pi = math.acos(-1)
	degree = (rad * 180) / pi
	return abs(degree)


# 벡터 안 요소들의 분산을 반환
def getVariance(v, mean):
	result = 0
	size = len(v)
	for i in range(size):
		diff = mean - v[i]
		result += (diff*diff)
	result /= size
	return result


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
	return result


# Find out and draw Hand Gesture
def drawHandGesture(original, mask):
	font = cv.FONT_HERSHEY_COMPLEX
	fontScale = 1
	color = (0, 255, 255)
	thickness = 1

	image, contours, hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		# 객체 외곽선 그리기
		cv.drawContours(original, [cnt], 0, (255, 0, 0), 3)  # blue

		# 손바닥 중심과 반지름 찾기
		(x, y), radius = cv.minEnclosingCircle(cnt)
		scale = 0.8
		center = (int(x), int(y))
		radius = int(radius)
		cv.circle(original, center, 2, (0, 255, 0), -1)
		cv.circle(original, center, int(radius*scale), (0, 255, 0), 2)

		# 손가락 개수를 세기 위한 원 그리기
		cImg = np.zeros(mask.shape, np.uint8)
		cv.circle(cImg, center, int(radius*scale), 255)

		# 원의 외곽선을 저장할 벡터
		image, circleContours, hierarchy = cv.findContours(cImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

		# 원의 외곽선을 따라 돌며 mask의 값이 0에서 1로 바뀌는 지점 확인
		points = []
		for i in reversed(range(1, len(circleContours[0]))):
			p1 = (circleContours[0][i][0][0], circleContours[0][i][0][1])
			p2 = (circleContours[0][i-1][0][0], circleContours[0][i-1][0][1])
			print(p1, p2)
			if mask[p1[1], p1[0]] == 0 and mask[p2[1], p2[0]] > 1:
				points.append(p1)

		# 손가락 마디(선) 그리기
		for i in range(len(points)):
			cv.line(original, center, points[i], (0, 255, 0), 3)

		# 제스처 출력
		text = getHandGesture(points)
		cv.putText(original, text, center, font, fontScale, color, thickness)

	return original


cap = cv.VideoCapture(0)
while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		# Invert left and right
		frame = cv.flip(frame, 1)
		# Skin detect and Noise reduction
		mask = detectSkin(frame)
		# Find out and Draw Hand Gesture
		detected = drawHandGesture(frame, mask)

		# Show window
		cv.namedWindow('Live', cv.WINDOW_NORMAL)
		cv.imshow('Live', detected)
		k = cv.waitKey(1) & 0xFF
		if k == 27:
			break
	else:
		print('error')

cap.release()
cv.destroyAllWindows()
