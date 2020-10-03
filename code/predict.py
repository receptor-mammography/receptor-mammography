#-*- coding: utf-8 -*-

# Libraries
import keras
import tensorflow
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from keras.optimizers import SGD, rmsprop
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.backend as K
import sys, glob
import os
import zipfile, io, re
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold


# Weight path
model_save_name = '../h5files/ER_VGG_best_performing.hdf5'
img_dir_test = '/path/to/test-img-dir'
dir_tf_log = '../tf_log'

classes = ["Negative", "Positive"]
file_ext = 'png'
image_size=300

# Image size
img_height=image_size
img_width=image_size

# The number of classes
num_classes = len(classes)

def create_model():
  global image_size
  input_shape = (image_size, image_size, 3)
  # Feature extractor
  from keras.applications.vgg16 import VGG16
  base_model = VGG16(weights= 'imagenet', include_top=False, input_shape=input_shape)
  # Classifier
  m = Sequential()
  m.add(Flatten(input_shape=base_model.output_shape[1:]))
  m.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
  m.add(BatchNormalization())
  m.add(Dropout(0.5))
  m.add(Dense(num_classes, activation='softmax'))
  predictions = m(base_model.output)

  model = Model(inputs=base_model.input, outputs=predictions)

  # Optimizer
  opt = SGD(lr=0.001)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def load_dataset(path, classes):
  global image_size
  X=[]
  Y=[]
  Files=[]
  for index, ClassLabel in enumerate(classes):
    ImagesDir = os.path.join(path, ClassLabel)
    # convert RGB
    print(index)
    print(ClassLabel)
    print(ImagesDir)
    files = glob.glob(ImagesDir+"/*."+file_ext)
    for i, file in enumerate(files):
        print(i)
        print(file)
        image = Image.open(file)
        image = image.convert('RGB')
        # Resize
        image = image.resize((image_size, image_size))
        # Image to array
        data = np.asarray(image)
        X.append(data)
        Y.append(index)
        Files.append(file)
  X = np.array(X)
  Y = np.array(Y)
  return (X, Y, np.array(Files))

Xtest, Ytest, testFiles = load_dataset(img_dir_test, classes)

print(Xtest.shape, Ytest.shape)

# Initialization
K.clear_session()
config = tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(allow_growth=True))
session = tensorflow.Session(config=config)
K.tensorflow_backend.set_session(session)

model = create_model()
model.summary()
model.load_weights(model_save_name)

# Prediction
# x: lists to be predicted
def predict(x):
    global model
    pred = model.predict(x)
    return np.array(pred)

print("Predict:")
for i in np.arange(num_classes):
    x = Xtest[Ytest == i]
    files = testFiles[Ytest == i]
    # normalization
    x = x / 255.
    
    result = predict(x)
    for k,file in enumerate(files):
       print("class ", i,"\tpredict ", np.argmax(result[k]), "\t", file, "\t", result[k]) 

