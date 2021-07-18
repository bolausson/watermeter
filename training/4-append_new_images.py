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
import argparse
import matplotlib.pyplot as plt
import pprint
from termcolor import colored, cprint
from colorama import Fore, Back, Style
import numpy as np
from tensorflow import keras
from datetime import datetime

#NEW_PICKLE_PATH_ONLY = "data_sorted.pickle"
#EXISTING_IMAGE_PICKLE = "nonde"
#NEW_PICKLE_PATH_ONLY = "data_sorted-retrain.pickle"
#EXISTING_IMAGE_PICKLE = "training_image-set_2021.03.19_16-45-11.pickle"


parser = argparse.ArgumentParser(description='Train the model')

parser.add_argument(
    '-n',
    '--new',
    dest='new_pickl_path_only',
    default='data_sorted-retrain.pickle',
    type=str,
    help='Pickle file containing path to new images',)

parser.add_argument(
    '-a',
    '--append_to',
    dest='existing_image_pickl',
    type=str,
    help='Pickle file to append the new images to')

args = parser.parse_args()

NEW_PICKLE_PATH_ONLY = args.new_pickl_path_only
EXISTING_IMAGE_PICKLE = args.existing_image_pickl

if not EXISTING_IMAGE_PICKLE:
    try:
        EXISTING_IMAGE_PICKLE = sorted(glob.glob("image-set/*.pickle"))[-1]
    except IndexError as e:
        print("No imagset given or found - creating new pickle!")
    else:
        print(f"Use {Fore.YELLOW}{EXISTING_IMAGE_PICKLE}{Style.RESET_ALL} to append now data?")
        res = input("y/n\n")
        if res != "y":
            sys.exit(0)

        cprint(f"Using {EXISTING_IMAGE_PICKLE} to append new data!", "red")

time.sleep(5)

image_size = (76, 36)
num_classes = 10
timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
pp = pprint.PrettyPrinter(indent=4)

if os.path.isfile(NEW_PICKLE_PATH_ONLY):
    print(f"Opening {NEW_PICKLE_PATH_ONLY}")
    with open(NEW_PICKLE_PATH_ONLY, 'rb') as f:
         data = pickle.load(f)
else:
    cprint(f"Couldn't find {NEW_PICKLE_PATH_ONLY}", "red")
    sys.exit(0)

print(f"Reading images into dictionary and resizing to {image_size}")

new_data = {}
new_count = {}
new_sum = 0
for i in range(0, 10, 1):
    new_data[i] = []
    for org in data.keys():
        for j in data[org][i]:
            #print(j)
            img = keras.preprocessing.image.load_img(j, color_mode="grayscale", target_size=image_size)
            img = keras.preprocessing.image.img_to_array(img)
            new_data[i].append(img)
    items = len(new_data[i])
    new_count[i] = items
    new_sum += items

cprint("Done reading files", "green")

print(f"Items for number in new dataset ({new_sum}):")
for key in new_count.keys():
    print(f"{key} --> {new_count[key]}")

if EXISTING_IMAGE_PICKLE:
    print(f"Opening {EXISTING_IMAGE_PICKLE}")
    
    with open(EXISTING_IMAGE_PICKLE, 'rb') as f:
         old_data = pickle.load(f)
    
#    old_count = {}
#    for i in old_data.keys():
#        old_count[i] = 0
#        for j in old_data[i]:
#            old_count[i] += 1

#    print("Items for number in old dataset:")
#    for key in old_count.keys():
#        print(f"{key} --> {old_count[key]}")

    print("Adding old data to new data")
    old_count = {}
    old_sum = 0
    merged_count = {}
    merged_sum = 0
    for i in range(0, 10, 1):
        try:
            old_list = old_data[i]
        except KeyError as e:
            old_list = []
        try:
            new_list = new_data[i]
        except KeyError as e:
            new_list = []
        
        new_data[i] = old_list + new_list
        old_items = len(old_list)
        old_count[i] = old_items
        old_sum += old_items
        merged_items = len(new_data[i])
        merged_count[i] = merged_items
        merged_sum += merged_items

    print(f"Items for number in old dataset ({old_sum}):")
    for key in old_count.keys():
        print(f"{key} --> {old_count[key]}")

    print(f"Items for number in merged dataset ({new_sum} + {old_sum} = {new_sum + old_sum} | {merged_sum}):")
    for key in merged_count.keys():
        print(f"{key} --> {merged_count[key]}")

NEW_IMAGE_PICKLE = f"image-set/training_image-set_{timestamp}.pickle"
print(f"Saving data to {NEW_IMAGE_PICKLE}")

with open(NEW_IMAGE_PICKLE, "wb") as f:
    pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)
