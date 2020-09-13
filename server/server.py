import os
from flask import send_file
from flask import Flask, render_template, request
from flask import Response
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
import cv2
import random
import csv

# module level variables ##############################################################################################
UPLOAD_FOLDER = os.getcwd() + '/upload_images'
SAVE_IMAGE_DIR = os.getcwd() + "/static/"
#######################################################################################################################

ALLOWED_EXTENSIONS = ['jpg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['myfile']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("starting program . . .")

            #if StrictVersion(tf.__version__) < StrictVersion('1.5.0'):
            #    raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')
            # end if

            imageFilePaths = []
            for imageFileName in os.listdir(UPLOAD_FOLDER):
                if imageFileName.endswith(".jpg"):
                    imageFilePaths.append(UPLOAD_FOLDER + "/" + imageFileName)
            
            image_path = imageFilePaths[-1]
            print(image_path)
            Image_Coin_Counter = cv2.imread(image_path)

            if Image_Coin_Counter is None:
                print("error reading file " + image_path)

            text = "안녕하세요."
            print("Final Result Text : " + text)
            imagefilename = SAVE_IMAGE_DIR + r'image.jpg'
            cv2.imwrite(imagefilename, Image_Coin_Counter)
        return text
if __name__ == "__main__":
   app.run(host="0.0.0.0", port="8080")