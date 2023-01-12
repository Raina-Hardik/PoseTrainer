from numpy import expand_dims, argmax
from os import listdir  
import cv2
from keras.models import Model, load_model
from text_to_speech import *
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img, img_to_array
from time import sleep
import argparse

def predictor(frame):
    frame = cv2.resize(frame, (300,300))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    x = img_to_array(img)
    x = expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)[0]
    pose = classifications[argmax(classes)]
    t2s(pose)

target = 0

def main():
    path = 'dataset/'                       # Reduced dataset
    classes = listdir(path)
    classifications = {i:class_name for i, class_name in enumerate(classes)}

    model = load_model('posetrainer.h5')

    cap = cv2.VideoCapture(target)
    #cap = cv2.VideoCapture(0)     
    count = 0
    while(cap.isOpened()):
        count += 1
        ret, frame = cap.read()
        if count == 500: 
            count=0
            frame = cv2.resize(frame, (300,300))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            x = img_to_array(img)
            x = expand_dims(x, axis=0)
            img_data = preprocess_input(x)
            classes = model.predict(img_data)[0]
            if(max(classes) <= 0.5): pass
            val = argmax(classes)
            print(val)
            pose = classifications[val]
            t2s(pose)
        else: 
            cv2.imshow('PoseTrainer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    target = 'poses.mp4'
    main()
