#-*- coding: utf-8 -*-

# Import libraries
import keras
import tensorflow

# Switch the model which you wanted to use
from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50 import ResNet50
#from keras.applications.densenet import DenseNet121

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

# If "True", you can predict without training algorithm. 
predict_only = True 

# If "True", you can use generator.
use_generator = True

# If "True", you can use cross-validation. 
use_cross_validation = True
# When you use cross_validation, decide how many validation should be divided.
n_splits=4

# If "True", you can use MNIST as trial.
use_mnist = False

# If "True", you can use downsample.
use_down_sampling=True

# Decide how many downsamples you use in each validation.
sub_split=10

# Decide batch_size and epochs
batch_size=4
epochs=50

# Definition of models
def create_model():
  global image_size
  input_shape = (image_size, image_size, 3)
  # Choose the model from "VGG16, InceptionV3, ResNet50, DenseNet121"
  base_model = VGG16(weights= 'imagenet', include_top=False, input_shape=input_shape)
  m = Sequential()
  m.add(Flatten(input_shape=base_model.output_shape[1:]))
  m.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
  m.add(BatchNormalization())
  m.add(Dropout(0.5))
  m.add(Dense(num_classes, activation='softmax'))
  predictions = m(base_model.output)

  model = Model(inputs=base_model.input, outputs=predictions)
  opt = SGD(lr=0.001)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

# Output directory
model_save_name = '../h5files/SGD_ensemble_%d.hdf5'
csvlog_name = '../log/SGD.log'

# Definition of downsampling
def down_sample(x,y, shuffle=True):
   global use_down_sampling
   if not use_down_sampling:
     r = np.arange(len(y))
     if shuffle: r=np.random.permutation(r)
     return r
   categories = np.unique(y)
   indices = {}
   for c in categories:
     indices[c] = np.where(y == c)[0]
   each_num = min([len(indices[i]) for i in indices])
   res = []
   for c in categories:
     index = np.random.choice(indices[c], each_num, replace=False)
     res.extend(index)
   if shuffle:
      res = np.random.permutation(res)
   print("size change ", len(y), ' -> ', len(res), '(min',each_num,')')
   return res
K.clear_session()

if use_mnist:
  img_dir = '/path/to/MNIST/Data/Train'
  img_dir_test = '/path/to/MNIST/Data/Validation'
  classes = ['0', '1']
  file_ext = 'jpg'
  image_size=32
else:
  img_dir = '/path/to/dataset'
  img_dir_test = '/path/to/test-dataset'
  classes = ["Negative", "Positive"]
  file_ext = 'png'
  image_size=500

# Set image size
img_height=image_size
img_width=image_size

# Set number of classes
num_classes = len(classes)

# Fetch dataset lists
def load_dataset(path, classes):
  global image_size
  X=[]
  Y=[]
  Files=[]
  for index, ClassLabel in enumerate(classes):
    global use_generator
    ImagesDir = os.path.join(path, ClassLabel)
    print(index)
    print(ClassLabel)
    print(ImagesDir)
    files = glob.glob(ImagesDir+"/*."+file_ext)
    for i, file in enumerate(files):
        print(i)
        print(file)
        image = Image.open(file)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)
        Files.append(file)
  X = np.array(X)
  Y = np.array(Y)
  if not use_generator:
    X = X.astype('float32') / 255
  return (X, Y, np.array(Files))
X, Y, _ = load_dataset(img_dir, classes)
Xtest, Ytest, testFiles = load_dataset(img_dir_test, classes)

print(X.shape, Y.shape)

dir_tf_log = '../tf_log'

# Initialization
config = tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(allow_growth=True))
session = tensorflow.Session(config=config)
K.tensorflow_backend.set_session(session)

# Augmentation
horizontal_flip = True
vertical_flip = True
if use_mnist:
  horizontal_flip = False
  vertical_flip = False
datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.9,1.1],
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        zoom_range=False,
        fill_mode='reflect')


# Callbacks
csv_logger = CSVLogger(csvlog_name)

# Training
script_path = os.path.abspath(__file__)

skf = KFold(n_splits=n_splits, shuffle=True)
index = 0
test_pred = []
if not predict_only:
	train_val = []
	if use_cross_validation:
		train_val = skf.split(X, Y)
	else:
		i = np.random.permutation(np.arange(len(X)))
		s = int(len(i) * 3 / 4) + 1
		train_val = [(i[0:s], i[s:])]
	for train, val in train_val:
		model = create_model()
		if index == 0: model.summary()
		model_name = model_save_name % index
		index += 1
		checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True)
		for i2 in np.arange(sub_split):
			print("begin split ",index,'/',n_splits,", subsplit ",i2+1,'/',sub_split)
			i = down_sample(X[train], Y[train])
			x, y = X[train][i], Y[train][i]

			y = [to_categorical(i, num_classes=num_classes) for i in y]
			
			i = down_sample(X[val], Y[val])
			xval, yval = X[val][i], Y[val][i]
			yval = [to_categorical(i, num_classes=num_classes) for i in yval]
			
			i = down_sample(Xtest, Ytest)
			xtest, ytest = Xtest[i], Ytest[i]
			ytest = [to_categorical(i, num_classes=num_classes) for i in ytest]
			x = np.array(x);       y = np.array(y)
			xval = np.array(xval); yval = np.array(yval)
			xtest = np.array(xtest); ytest = np.array(ytest)
			if use_generator:
				history = model.fit_generator(
					    datagen.flow(x, y, batch_size = batch_size),
	        		            steps_per_epoch=x.shape[0] // batch_size,
	        		            epochs=epochs,
	        		            validation_data = datagen.flow(xval, yval, batch_size = batch_size),
	        		            validation_steps=xval.shape[0] // batch_size,
	        		            verbose=1,
	        		            callbacks=[csv_logger, checkpointer])
			else:
				history = model.fit(x=x, y=y, batch_size=batch_size,
					epochs=epochs, verbose=1,
					validation_data = (xval, yval),
					callbacks=[csv_logger, checkpointer])
		# Evaluation
		if use_generator:
			scores = model.evaluate_generator(datagen.flow(xtest, ytest, batch_size=batch_size), verbose=0)
		else:
			scores = model.evaluate(x=xtest, y=ytest, batch_size=batch_size, verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		test_pred.append(scores[1])
		K.clear_session()

preds=[]
models=[]

# Prediction
# x: list for prediction
# If "raw=True", the categorical cross-entropy prediction value for each image are provided. If "raw=False", present summary only.
# model_range: Choice which h5file you use.
def predict(x, raw=False, model_range=None):
    global model_save_name, n_splits, preds, use_cross_validation, predict_only
    pred = []
    default_model_range = np.arange(n_splits if use_cross_validation else 1)
    if len(models) == 0:
      for i in default_model_range:
          model = create_model()
          model.load_weights(model_save_name % i)
          models.append(model)
      if predict_only:
          models[0].summary()

    model_range = np.array(model_range) if model_range else default_model_range
    for i in model_range:
        model = models[i]
        p = model.predict(x)
        pred.append(p)
    if not raw:
      pred = np.array(pred)
      pred = np.sum(pred, axis=0)
    preds.append(pred)
    return np.array(pred)

print("Predict:")
for i in np.arange(num_classes):
    x = Xtest[Ytest == i]
    files = testFiles[Ytest == i]
    if use_generator:
       x = x / 255.
    
    result = predict(x)
    #result = predict(x, model_range=[1])
    for k,file in enumerate(files):
       print("class ", i,"\tpredict ", np.argmax(result[k]), "\t", file, "\t", result[k]) 
#K.clear_session()

