import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img
# import cupy as cp
from random import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow.keras.backend as K 
from keras.models import load_model

def getAuthor(p,labels):
    print("------------------------",p[0])
    max1=p[0]
    count=0
    for i in p:
        if i>max1:
            max1=i 
            index=count
        count+=1
    return labels[index],max1


print(keras.__version__)
# model = load_model("model.h5")
file1 = open('labels.txt', 'r') 
count = 0
labels=[]
while True: 
    count += 1 
    line = file1.readline()  
    if not line: 
        break
    labels.append(line.strip())
file1.close() 
# print(labels)
image_raw = cv2.imread("a01-049x-s01-01.png", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image_raw, (1000, 100))
img_array = img_to_array(image)
img_batch = np.expand_dims(img_array, axis=0)
model =tf.keras.models.load_model("model.h5")
p=model.predict(img_batch)
for i in p:
    predictions=i.tolist()
auth,probability=getAuthor(predictions,labels)
print(auth,probability)