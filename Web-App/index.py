from flask import Flask
from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf 
from tqdm import tqdm
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, array_to_img
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


app = Flask(__name__)
UPLOAD_FOLDER = '/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
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
      image_raw = cv2.imread(f.filename, cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image_raw, (1000, 100))
      img_array = img_to_array(image)
      img_batch = np.expand_dims(img_array, axis=0)
      model =tf.keras.models.load_model("model.h5")
      p=model.predict(img_batch)
      for i in p:
        predictions=i.tolist()
      auth,probability=getAuthor(predictions,labels)
      Rhtml="<html><head><!--Google Fonts--><link rel='preconnect' href='https://fonts.gstatic.com'><link href='https://fonts.googleapis.com/css2?family=Goldman&display=swap' rel='stylesheet'><link rel='preconnect' href='https://fonts.gstatic.com'><link href='https://fonts.googleapis.com/css2?family=Alex+Brush&display=swap' rel='stylesheet'><style>body{    font-family: 'Alex Brush', cursive;background-color:#86c1df;}.result{margin-top: 300px;text-align: center;}p{font-family: 'Goldman', cursive;}</style></head><body><div class='result'><h1>The Author of this handwriting can be author with author ID<p>"+str(auth)+"</p></h1> <br><h1>with a probability of<p>"+str(probability)+"</p></h1></div></body></html>"
      return Rhtml
app.run()