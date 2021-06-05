#!/usr/bin/env python3
#
#

import os
import sys
import cv2
import glob
import time
import copy
import pprint
import pickle
import shutil
import argparse
import numpy as np
import collections
import itertools
from datetime import datetime
from itertools import zip_longest
from termcolor import colored, cprint
from colorama import Fore, Back, Style

TRAININGSET = "./retrain_images"
DATAPATH_PICKLE = "data_path-retrain.pickle"
SORTED_PICKLE = "data_sorted-retrain.pickle"
timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
SORTED_PICKLE_BACKUP = SORTED_PICKLE.replace(".pickle", f"_before-modification_{timestamp}.pickle")

cv2keydict = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8, 57: 9}

parser = argparse.ArgumentParser(description='Fix wrong value asignment')

parser.add_argument(
    '-i',
    '--image',
    dest='image_to_fix',
    type=str,
    help='Image for which the value should be changed')

parser.add_argument(
    '-n',
    '--number',
    dest='value_for_image_to_fix',
    default=False,
    type=int,
    help='Actual value for the image')

args = parser.parse_args()

IMAGE_TO_FIX = args.image_to_fix
VLAUE_FOR_IMAGE_TO_FIX = args.value_for_image_to_fix

pp = pprint.PrettyPrinter(indent=4)

if os.path.isfile(DATAPATH_PICKLE):
    print(f"Opening {DATAPATH_PICKLE}")
    with open(DATAPATH_PICKLE, 'rb') as f:
        original = pickle.load(f)
else:
    print(f"File {DATAPATH_PICKLE} not found")
    sys.exit()

if os.path.isfile(SORTED_PICKLE):
    print(f"Opening {SORTED_PICKLE}")
    with open(SORTED_PICKLE, 'rb') as sorted:
         data = pickle.load(sorted)
else:
    print(f"File {SORTED_PICKLE} not found")

modified_data = copy.deepcopy(data)

#pp.pprint(data)
items = data.items()

def finder():
    for h, i in data.items():
        testh = h
        for j, k in i.items():
            item_of_interest = [x for x in k if IMAGE_TO_FIX in x]
            if item_of_interest:
                item_of_interest = item_of_interest[0]
                image_to_fix_list_pos, image_to_fix_full_path = [[l, m] for l, m in enumerate(k) if m == item_of_interest][0]
                original_key = h
                current_value_key = j
                return item_of_interest, image_to_fix_list_pos, image_to_fix_full_path, original_key, current_value_key

item_of_interest, image_to_fix_list_pos, image_to_fix_full_path, original_key, current_value_key = finder()

print(item_of_interest)
print(IMAGE_TO_FIX)

x = 0
y = 100
win = "Feature"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(win, 36, 76)
cv2.moveWindow(win, x,y)
im = cv2.imread(item_of_interest)
cv2.imshow(win, im)

print(f"Value assigned to image: {Fore.YELLOW}{current_value_key}{Style.RESET_ALL}")

if VLAUE_FOR_IMAGE_TO_FIX:
    print(f"Assigning value {VLAUE_FOR_IMAGE_TO_FIX} to shown image (y/n)\n")
else:
    print("Enter value which should be assigned to the shown image")

keypress = cv2.waitKey(0)

if keypress == 27:
    # ESC key
    cv2.destroyAllWindows()
    sys.exit(0)
elif keypress == 121:
    # y key
    print(f"New value assigned to image: {Fore.GREEN}{VLAUE_FOR_IMAGE_TO_FIX}{Style.RESET_ALL}")
elif keypress == 110:
    # n key
    print("Terminating")
    sys.exit(0)
elif keypress in cv2keydict.keys():
    NEW_VLAUE_FOR_IMAGE_TO_FIX = cv2keydict[keypress]
    print(f"New value assigned to image: {Fore.GREEN}{NEW_VLAUE_FOR_IMAGE_TO_FIX}{Style.RESET_ALL}")
else:
    print(f"Unknow key {keypress} - terminating!")
    cv2.destroyAllWindows()
    sys.exit(0)

cv2.destroyAllWindows()

print(f"Original:{Fore.YELLOW}")
pp.pprint(data[original_key])

modified_data[original_key][current_value_key].remove(image_to_fix_full_path)
modified_data[original_key][NEW_VLAUE_FOR_IMAGE_TO_FIX].append(image_to_fix_full_path)

print(f"{Style.RESET_ALL}Modified:{Fore.GREEN}")
pp.pprint(modified_data[original_key])
print(Style.RESET_ALL)

resp = input(f"{Fore.RED}Overwrite {SORTED_PICKLE}?\n(y/n)\n{Style.RESET_ALL}")

if resp == "y":
    with open(SORTED_PICKLE_BACKUP, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    with open(SORTED_PICKLE, "wb") as f:
        pickle.dump(modified_data, f, pickle.HIGHEST_PROTOCOL)
else:
    print("Discarding changes")
