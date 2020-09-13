# import the necessary packages
from imutils.video import VideoStream
import imutils
import time
import os
import cv2
import requests as req
import numpy as np
import io
from PIL import Image

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    # frame = imutils.resize(frame, width=1100)
    frame = cv2.flip(frame, 1)

    # 서버에 업로드
    filename = "image.jpg"
    if not cv2.imwrite(filename, frame):
        raise RuntimeError("Unable to capture image")
    files = {'myfile': open(filename, 'rb')}
    # url = "http://52.21.196.66:8080/upload"
    url = "http://192.168.19.145:8080/upload"
    response = req.post(url, files=files)
    result = response.text
    print(result)

    # 서버에서 다운로드
    imgurl = "http://192.168.19.145:8080/static/image.jpg"
    img_data = req.get(imgurl).content
    print(type(img_data))
    print(img_data)
    # image = np.array(Image.open(io.BytesIO(img_data)))
    encoded_img = np.fromstring(img_data, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    # mat = np.asarray(img_data, dtype=np.uint8)
    # with open(os.getcwd() + '/static/cvimage.jpg', 'wb') as handler:
    #    handler.write(img_data)

    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
