##############################################
# Author: Jakub Svoboda
# Email:  xsvobo0z@stud.fit.vutbr.cz
# School: Brno University of Technology
##############################################
# Test script for evaluation on LFW dataset.
# The aligned dataset will be automatically downloaded,
# as well as the pairs.txt file.
##############################################


# Imports
from google_drive_downloader import GoogleDriveDownloader as gdd
import tensorflow as tf
import os
import wget
from external.evaluate_lfw import *
from network import getNetwork

# Hyperparameters and constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 224                             # 224 for mobilenet, 299 for InceptionV3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
DIR = 'CASIA-WebFace_mtcnnpy'
LFW_PAIRS_PATH = r'pairs.txt'
LFW_DIR = r'lfw_mtcnn'
BACKBONE = 'MobileNetV3Large'   
CKPT_DIR = os.path.join('./checkpoints')    # Best .hdf5 model storage

# Download the cropped LFW dataset
if not os.path.exists('lfw.zip'):
    gdd.download_file_from_google_drive(file_id='1R2wJDuZIxu1Rtx4Ei05cbokUTP0EV2YQ', dest_path='./lfw.zip', unzip=True)

# Get pairs.txt
if not os.path.exists('pairs.txt'):
    wget.download('http://vis-www.cs.umass.edu/lfw/pairs.txt')

model = getNetwork(backbone = BACKBONE, embeddingSize=128, fcSize=1024, l2Norm=True)
model.load_weights('./checkpoints/mobile_9708.hdf5')
evaluate_LFW(model,128,use_flipped_images=False,distance_metric=1,verbose=2,N_folds=10)

