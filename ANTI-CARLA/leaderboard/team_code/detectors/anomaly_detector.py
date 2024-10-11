import cv2
import numpy as np
#from keras.models import Model, model_from_json
#import numpy as np
#from keras import backend as K
#import tensorflow as tf
import os
from sklearn.utils import shuffle
#from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
import csv

def occlusion_detector(image,threshold):
    """Determines occlusion percentage and returns
       True for occluded or False for not occluded"""

    # Create mask and find black pixels on image
    # Color all found pixels to white on the mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[np.where((image <= [15,15,15]).all(axis=2))] = [255,255,255]

    # Count number of white pixels on mask and calculate percentage
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    percentage = (cv2.countNonZero(mask)/ (w * h)) * 100
    #print("occlusion%f"%percentage)
    if percentage > threshold:
        return percentage, 1
    else:
        return percentage, 0

#image = cv2.imread('2.jpg')
#percentage, occluded = detect_occluded(image)
#print('Pixel Percentage: {:.2f}%'.format(percentage))
#print('Occluded:', occluded)

def blur_detector(image, threshold=20):
    """
    Determines if an image is blur
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    #print("Blur%f"%fm)
    if fm < threshold:
        return fm, 1
    else:
        return fm, 0
