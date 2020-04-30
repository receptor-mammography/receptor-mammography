# Classification of receptor expression in breast cancer from mammography
This algorthm was developed to classify receptor expressions in breast cancer from mammograms.

You can use this algorithm after downloaded hdf5 file from release page. This algorithm is not for a diagnostic use.

## Install
Clone this repository.

Download a hdf5 file (*_best-performing.hdf5) into the directory of "h5files".

hdf5 file can be downloaded from: https://github.com/receptor-mammography/receptor-mammography/releases

## Usage
python SGD.py in the directory of "code".

Default model was VGG16. If you change the base_model part in the SGD.py, you can try different models. 

The hdf5files we have prepared are VGG16, InceptionV3, ResNet50, and DenseNet121.

## Enviroment
This algorithm was built in the TensorFlow framework (https://www.tensorflow.org/) with the Keras wrapper library (https://keras.io/).

tensorflow-gpu 1.10.0

Keras 2.2.4

## Research paper
This algorithm was published on XXX.
