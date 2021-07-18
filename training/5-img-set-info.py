#!/usr/bin/env python3.8
#
#

import os
import cv2
import sys
import glob
import pickle
import random
import argparse
import pprint
from termcolor import colored, cprint
from colorama import Fore, Back, Style
#from tensorflow import keras

#NEW_PICKLE_PATH_ONLY = "data_sorted.pickle"
#EXISTING_IMAGE_PICKLE = "nonde"
#NEW_PICKLE_PATH_ONLY = "data_sorted-retrain.pickle"
#EXISTING_IMAGE_PICKLE = "training_image-set_2021.03.19_16-45-11.pickle"


parser = argparse.ArgumentParser(description='Get info about image-set')

parser.add_argument(
    '-i',
    '--imgset',
    dest='imgset',
    type=str,
    help='Pickle file containing image set',)

args = parser.parse_args()

IMGSET = args.imgset

image_size = (76, 36)
num_classes = 10

pp = pprint.PrettyPrinter(indent=4)

if IMGSET:
    if os.path.isfile(IMGSET):
        IMGSETLIST = [IMGSET]
    else:
        cprint(f"Couldn't find {IMGSET}", "red")
        sys.exit(0)
else:
    IMGSETLIST = sorted(glob.glob("image-set/*.pickle"))

overall_count = {}
previous_count = False
for c, IMGS in enumerate(IMGSETLIST):
    print(f"{IMGS}")
    with open(IMGS, 'rb') as f:
        data = pickle.load(f)

    if c == 0:
        print("Data structure")
        pprint.pprint(data, depth=1)
        print("----------------")

    count = []
    for i in range(0,10,1):
        items = len(data[i])
        #print(f"{i: >2}: {items}")
        #img = keras.preprocessing.image.load_img(j)
        #img = keras.preprocessing.image.img_to_array(img)
        count.append(items)
        try:
            overall_count[i] += items
        except KeyError as e:
            overall_count[i] = items
    
    SUM = sum(count)
    l = len(str(SUM))
    
    if previous_count:
        for i, j in enumerate(count):
            print(f"{Fore.CYAN}{i: >3}{Style.RESET_ALL}: {Fore.YELLOW}{j: >{l}}{Style.RESET_ALL} (+{j - previous_count[i]})")
        
        print(f"SUM: {sum(count)}")
    else:
        for i, j in enumerate(count):
            print(f"{Fore.CYAN}{i: >3}{Style.RESET_ALL}: {Fore.YELLOW}{j: >{l}}{Style.RESET_ALL}")
        
        print(f"SUM: {sum(count)}")
    previous_count = count

cprint("==============SUM==================", "green")
SUM = sum(overall_count[i] for i in overall_count.keys())
l = len(str(SUM))
for key in overall_count.keys():
    print(f"{Fore.CYAN}{key: >3}{Style.RESET_ALL}: {Fore.YELLOW}{overall_count[key]: >{l}}{Style.RESET_ALL}")

print(f"SUM: {SUM}")
