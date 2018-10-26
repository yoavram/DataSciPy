import keras
from tensorflow.examples.tutorials.mnist import input_data
from urllib.request import urlretrieve
import os
import zipfile

# MNIST and Fasihon MNIST, day 2 and 3
print('* MNIST (Keras)...')
keras.datasets.mnist.load_data()
print('* Fashion-MNIST (Keras)...')
keras.datasets.fashion_mnist.load_data()
print('* MNIST (TensorFlow)...')
input_data.read_data_sets('MNIST_data', one_hot=True)

# ResNet50, day 3
print('ResNet50')
keras.applications.resnet50.ResNet50(weights='imagenet')

# ESC-50, day 3
url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
fname = 'data/ESC.zip'
if not os.path.exists(fname):
	print('* ESC-50...')
	urlretrieve(url, fname)
	with zipfile.ZipFile(fname) as z:
		z.extractall('data/')

# SpeechEmotion, day 3
url = 'https://github.com/yoavram/SpeechEmotion/archive/master.zip'
fname = 'data/SpeechEmotion.zip'
if not os.path.exists(fname):
	print('* SpeechEmotion...')
	urlretrieve(url, fname)
	with zipfile.ZipFile(fname) as z:
		z.extractall('data/')
	
# conda install librosa # for day 3