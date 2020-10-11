# try:
#     import keras
# except ModuleNotFoundError:
#     from tensorflow import keras
import requests
import os
import shutil
import zipfile
requests.packages.urllib3.disable_warnings()

def download_file(url, fname):
    r = requests.get(url, stream=True, verify=False)
    with open(fname, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

# MNIST and Fasihon MNIST, day 2 and 3
# print('* MNIST (Keras)...')
# keras.datasets.mnist.load_data()
# print('* Fashion-MNIST (Keras)...')
# keras.datasets.fashion_mnist.load_data()

# ResNet50, day 3
# print('* ResNet50...')
# keras.applications.resnet50.ResNet50(weights='imagenet')

# ESC-50, day 3
url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
fname = 'data/ESC.zip'
print('* ESC-50...')
if not os.path.exists(fname):
	download_file(url, fname)
try:
    with zipfile.ZipFile(fname) as z:
        z.extractall('data/')
except zipfile.BadZipFile:
    print("Problem with file {}, delete it and try again".format(fname))

# SpeechEmotion, day 3
url = 'https://github.com/yoavram/SpeechEmotion/archive/master.zip'
fname = 'data/SpeechEmotion.zip'
print('* SpeechEmotion...')
if not os.path.exists(fname):
	download_file(url, fname)
with zipfile.ZipFile(fname) as z:
    z.extractall('data/')

