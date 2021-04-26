##############################################
# Author: Jakub Svoboda
# Email:  xsvobo0z@stud.fit.vutbr.cz
# School: Brno University of Technology
##############################################
# Crops the images in the DIR folder with MTCNN
# The cropped image should have MARGIN pixels around the face in each direction.
# If the original image does not have sufficient pixels on given side, the side of the original image
# is selected as the new margin border.
# It is significantly faster to download the premade cropped datasets (see train script) 
##############################################

import cv2
import mtcnn
import os
from glob import glob
import numpy as np
import pathlib

DIR = os.path.join('..', 'lfw')
CROPPEDDIR = os.path.join('..', 'lfw_cropped')
MARGIN = 32

# Create new folder for the cropped images, must be done level wise otherwise windows api fails
if not os.path.isdir(CROPPEDDIR):
    os.mkdir(CROPPEDDIR)

# init mtcnn detector
detector = mtcnn.MTCNN()

# Get all images, for LFW the format is .jpg 
images = glob(os.path.join(DIR, '*/*.jpg'))


for imgPath in images:
    img = cv2.imread(imgPath)              # Load image
    result = detector.detect_faces(img)    # Pass through mtcnn
    if len(result) > 0:                    # only crop if there is exactly one face fount
        bounding_box = result[0]['box']
        x = np.max(bounding_box[0]-(MARGIN//2),0)
        y = np.max(bounding_box[1]-(MARGIN//2),0) 
        x2 = np.minimum(bounding_box[0]+bounding_box[2]+(MARGIN//2),img.shape[0])
        y2 = np.minimum(bounding_box[1]+bounding_box[3]+(MARGIN//2),img.shape[1])
        img = img[y:y2, x:x2]              # Crop
        print("Successfull crop", imgPath)
    else:
        print('Cant crop', imgPath)

    # Create new folder in CROPPEDDIR    
    if not os.path.isdir(os.path.join(CROPPEDDIR, pathlib.PurePath(imgPath).parent.name)):    
        os.mkdir(os.path.join(CROPPEDDIR, pathlib.PurePath(imgPath).parent.name))

    # Save image
    if img.size > 20:        # MTCNN can return invalid coordinates, save original when this happens
        cv2.imwrite(os.path.join(CROPPEDDIR, pathlib.PurePath(imgPath).parent.name, pathlib.PurePath(imgPath).name), img)
    else:
        img = cv2.imread(imgPath)  
        cv2.imwrite(os.path.join(CROPPEDDIR, pathlib.PurePath(imgPath).parent.name, pathlib.PurePath(imgPath).name), img)
