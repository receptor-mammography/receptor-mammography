#-*- coding: utf-8 -*-

# Libraries
import keras
import tensorflow
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
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


# If "True", predict with generator. If "False", predict without generator.
use_generator = True

# Weight path
model_save_name = '../h5files/ER_VGG_best_performing.hdf5'
csvlog_name = '../log/SGD.log'
img_dir_test = '/path/to/test-img-dir'

classes = ["Negative", "Positive"]
file_ext = 'png'
image_size=300

# Image size
img_height=image_size
img_width=image_size

# The number of classes
num_classes = len(classes)

def load_dataset(path, classes):
  global image_size
  X=[]
  Y=[]
  Files=[]
  for index, ClassLabel in enumerate(classes):
    global use_generator
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
  # Convert data type and normalization
  if not use_generator:
    X = X.astype('float32') / 255
  return (X, Y, np.array(Files))

Xtest, Ytest, testFiles = load_dataset(img_dir_test, classes)

print(Xtest.shape, Ytest.shape)

dir_tf_log = '../tf_log'

# Initialization
K.clear_session()
config = tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(allow_growth=True))
session = tensorflow.Session(config=config)
K.tensorflow_backend.set_session(session)

# Augmentation
horizontal_flip = True
vertical_flip = True
datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.1,
        brightness_range=[0.9,1.0],
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        zoom_range=False,
        #channel_shift_range=30,
        fill_mode='reflect')

def create_model():
  global image_size
  input_shape = (image_size, image_size, 3)
  # Feature extractor
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
  # #FineTuning
  # for layer in model.layers[:14]:
  #   layer.trainable = False
  # for layer in model.layers[14:]:
  #   layer.trainable = True
  # model.summary()

  # Optimizer
  opt = SGD(lr=0.001)
  #opt = rmsprop(lr=5e-7, decay=5e-5)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = create_model()
model.summary()
model.load_weights(model_save_name)

# Prediction
# x: lists to be predicted
def predict(x, use_argmax=True):
    global model

    p = model.predict(x)
    pred = []
    pred.append(p)
    pred = np.array(pred)
    pred = np.sum(pred, axis=0)
    if use_argmax:
      pred = np.argmax(pred, axis=1)
    return np.array(pred)

print("Predict:")
for i in np.arange(num_classes):
    x = Xtest[Ytest == i]
    files = testFiles[Ytest == i]
    # If use generator, normalization is needed. 
    if use_generator:
       x = x / 255.
    
    result = predict(x, use_argmax=False)
    for k,file in enumerate(files):
       print("class ", i,"\tpredict ", np.argmax(result[k]), "\t", file, "\t", result[k]) 

