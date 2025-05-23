import tensorflow 
from tensorflow import keras
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--image_path',type=str,help='provide image path')
    arg=parser.parse_args()
    image_path=arg .image_path

    vgg16=keras.applications.VGG16()

    image=keras.utils.load_img(image_path,target_size=(224,224,3))
    input_arr=keras.utils.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=vgg16.predict(input_arr)
    print(np.argmax(predictions))