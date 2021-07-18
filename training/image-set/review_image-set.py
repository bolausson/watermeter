#!/usr/bin/env python3.8
#
#

import os
import cv2
import sys
import glob
import time
import pickle
import random
import pprint
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from termcolor import colored, cprint
from colorama import Fore, Back, Style

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

pp = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(description='Review image data')

parser.add_argument(
    '-i',
    '--input',
    dest='trainingdata_images',
    default='',
    type=str,
    help='Pickle file containing the images')

parser.add_argument(
    '-n',
    '--number',
    dest='num_to_review',
    default=False,
    type=int,
    help='Number to review (0-9)')

args = parser.parse_args()

#TRAININGDATA = "training-data-2021.03.30_19-58-48.pickle"
TRAININGDATA_IMAGES = args.trainingdata_images
NUM_TO_REVIEW = args.num_to_review

timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

if not TRAININGDATA_IMAGES:
    TRAININGDATA_IMAGES = sorted(glob.glob("./*.pickle"))[-1]
    print(f"Review {Fore.CYAN}{TRAININGDATA_IMAGES}{Style.RESET_ALL}?")
    res = input("y/n\n")
    if res != "y":
        sys.exit()

if os.path.isfile(TRAININGDATA_IMAGES):
    print(f"Opening {TRAININGDATA_IMAGES}")
    time.sleep(5)
    with open(TRAININGDATA_IMAGES, 'rb') as f:
        digit_dict = pickle.load(f)
else:
    sys.exit(1)

cv2.namedWindow("Feature")
x = 570
y = 1700
cv2.moveWindow("Feature", x,y)

if NUM_TO_REVIEW:
    review = [NUM_TO_REVIEW]
else:
    review = range(0, 10, 1)

for i in review:
    items = len(digit_dict[i])
    print(f"{items} images for number {i}")
    time.sleep(5)
    for j, k in enumerate(digit_dict[i]):
        print(f"{items - j}: {i}")
        
        #img = keras.preprocessing.image.load_img(j)
        #img = keras.preprocessing.image.img_to_array(img)
        cv2.imshow("Feature", k)
        keypress = cv2.waitKey(100)

        if keypress == 27:
            # ESC key
            cv2.destroyAllWindows()
            sys.exit(0)

        if keypress == 32:
            # SPACE key
            cv2.waitKey(0)

        if keypress == 115 or keypress == 110:
            # s key or n key or space key
            break

cv2.destroyAllWindows()
