import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
# import cupy as cp
from random import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow.keras.backend as K 
print(keras.__version__)
print("model now running ...")
cuda_training = input("Use CUDA enabled device y/N? (y if model to be trained on GPU)")

if cuda_training == 'N':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_root_path = './Data/'
data_path = './Data/Sentences/'
img_names = os.listdir(data_path)   

def EncodeLabels(labelsList, labels):
    Encoded_labels = list()
    for i in labels:
        encoded = np.zeros(505)
        j = labelsList.index(i)
        encoded[j] = 1
        Encoded_labels.append(encoded)
    return Encoded_labels


def label_extract(imageNameList):
    labels = list()
    for i in tqdm(range(len(imageNameList))):
        author_id = imageNameList[i][4:8]
        if author_id not in labels:
            labels.append(author_id)
    return labels 
   

def ImageSeggregator(imageNameList, labelList):
    dataPathClassList = list()
    for i in tqdm(labelList):
        for j in imageNameList:
            if i in j:
                dataPathClassList.append([j,i])
    return dataPathClassList

def DataLoaderOne(imageNameList,labelList,count):
    c = 1
    data = list()
    for i in tqdm(imageNameList):
        image_raw = cv2.imread(os.path.join(data_path,i), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image_raw, (1000, 100))
        for j in labelList: 
            if j in i:
                authorName = 'author' + str(labelList.index(j))
                data.append([image,authorName,j])
        c +=1
        if c>count:
            break
    return data

def DataLoaderTwo(labelList,count):
    c = 1
    data = list()
    seggregatedList = ImageSeggregator(img_names,Labels)
    for i in tqdm(seggregatedList):
        image_raw = cv2.imread(os.path.join(data_path,i[0]), cv2.IMREAD_GRAYSCALE)
        image = image = cv2.resize(image_raw, (1000, 100))
        authorName = 'author' + str(labelList.index(i[1]))
        data.append([image,authorName,i[1]])
        c += 1
        if c>count:
            break
    return data


def Augumentor(data,size):
    datagen = ImageDataGenerator(width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.0, vertical_flip = False,horizontal_flip = False, fill_mode = 'reflect')
    count = 0
    for i in tqdm(range(len(data))):
        image = data[i][0]
        shape = (100,1000,1)
        image = image.reshape( ( 1, ) + shape )
        j = 2
        prefix = data[i][1] + '-' + data[i][2] + '#'
        for k in datagen.flow(image, batch_size=32,shuffle = False, save_to_dir=os.path.join(data_root_path,'Augumented/'),save_prefix =prefix,save_format='png'):
            if j> size:
                break
            j += 1
        count += 1
        # if count>=10:
        #     break
        

def AugumentedDataLoader():
    aug_data = list()
    aug_path = os.path.join(data_root_path, 'Augumented')
    for img in tqdm(os.listdir(aug_path)):
        index_dash = img.index('-')
        index_hash = img.index('#')
        authorName = img[:index_dash]
        authorID = img[index_dash+ 1 : index_hash]
        image_raw = cv2.imread(os.path.join(aug_path,img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image_raw,(1000,100))
        aug_data.append([image,authorName,authorID])
    return aug_data

def FeatureLabelSplit(data):
    feature = list()
    label = list()
    authorIDlist = list()
    for i in data:
        a,b,c = i
        feature.append(a)
        label.append(b)
        authorIDlist.append(c)

    return feature,label,authorIDlist

print("The total number of images are = ", len(img_names))   
Labels  = label_extract(img_names)    
# f = open("labels.txt", "x")
# for i in Labels:
#     f.write(str(i))
#     f.write("\n")
# f.close()
# print("------------------------------file ready--------------------------------")
print("The total number of distinct classes are = ", len(Labels))
datapoints_per_class = int(len(img_names)/len(Labels))
print("The number of images per author are = ", datapoints_per_class)
a = input("Pick a loading method one/two!  ")
if a == 'one':
    dataset = DataLoaderOne(img_names,Labels,count = 3000)
else:
    dataset = DataLoaderTwo(Labels,count = 3000)
print(dataset[0][0].shape)
plt.imshow(dataset[0][0], cmap='gray')
plt.show()
print(dataset[700][0].shape)
plt.imshow(dataset[700][0], cmap='gray')
plt.show()

b = input("Create aygumented data? (y/N)")
if b == 'y':
    Augumentor(dataset,10)
aug_dataset = AugumentedDataLoader()
dataset_complete = dataset + aug_dataset
shuffle(dataset_complete)
X,z,y = FeatureLabelSplit(dataset_complete)
print("The total number of images in the final dataset are = ", len(dataset_complete))
X = np.array(X).reshape(-1,100,1000,1)
for i in X:
    i = i/255.0
y = EncodeLabels(Labels,y)
y = np.array(y)

del Labels
del img_names
del dataset
del dataset_complete
del aug_dataset
del z
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same',input_shape = (100,1000, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2),padding='same'))
model.add(Flatten())
model.add(Dense(505))
model.add(Activation('softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(X,y,batch_size = 32, epochs=20, validation_split=0.1)
model.save('./modelexp.h5')
