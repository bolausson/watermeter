#!/usr/bin/env python3
#
#

import os
import sys
import cv2
import glob
import time
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

parser = argparse.ArgumentParser(description='Check the categorized data.\nUse "space" to pause, "s" to adjust speed (ms), "Esc" to quit. User "-v" to get the path to the displayed images. You can use "display" form imagemagick to display images on the commandline')

parser.add_argument(
    '-n',
    '--number',
    dest='number_to_check',
    type=int,
    default=-1,
    help='Number to manually check')

parser.add_argument(
    '-s',
    '--speed',
    dest='wait_before_next_img',
    default=1,
    type=int,
    help='Wait x ms before showing the next number')

parser.add_argument(
    '-w',
    '--windows',
    dest='num_windows',
    default=10,
    type=int,
    help='Number of windos to displays images in')

parser.add_argument(
    '-v',
    '--verbose',
    dest='verbose',
    action='store_true',
    help='Show image names')

parser.add_argument(
    '-i',
    '--index',
    dest='start_with_index',
    default=0,
    type=int,
    help='Start looking through images from index onwards')

parser.add_argument(
    '-e',
    '--end',
    dest='end_with_index',
    type=int,
    help='Stop looking through images at index')

args = parser.parse_args()

NUMBER_TO_CHECK = args.number_to_check
WAIT_BEFORE_NEXT_IMG = args.wait_before_next_img
NUM_WINDOWS = args.num_windows
VERBOSE = args.verbose
START_WITH_INDEX = args.start_with_index
END_WITH_INDEX = args.end_with_index


cv2keydict = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8, 57: 9}

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

# count features per position
feature_count = {}
for woc in ["cold", "warm"]:
    feature_count[woc] = {}
    for pos in range(0, 7, 1):
        feature_count[woc][pos] = 0
        for f in original[woc].keys():
            try:
                samples = original[woc][f]["feature"][pos]
            except KeyError as e:
                pass
            else:
                num_samples = len(samples)
                feature_count[woc][pos] += 1

if NUMBER_TO_CHECK >= 0:
    CHECK = [NUMBER_TO_CHECK]
else:
    CHECK = list(range(0, 10, 1))

new_dic = {}
for grp in CHECK:
    new_dic[grp] = []
    for f in data.keys():
        for fet_image in data[f][grp]:
            fpath, fname = os.path.split(fet_image)
            new_dic[grp].append(fet_image)

for g in new_dic.keys():
    print(f'{g} -> {len(new_dic[g])}')

windows = []
x = 0
y = 100
for i in range(0, NUM_WINDOWS, 1):
    win = f"Feature{i}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(win, 36, 76)
    cv2.moveWindow(win, x,y)
    y += 150

check_dic = {}
for grp in CHECK:
    check_dic[grp] = new_dic[grp][START_WITH_INDEX:END_WITH_INDEX]
    n = len(check_dic[grp])
    m = n
    print(f"Next number: {grp}")
    for i, img in enumerate(check_dic[grp]):        
        win_num = 0
        win = f"Feature{win_num}"

        if VERBOSE:
            msg = f"{n}/{m} (Index: {i} | Original Index: {i + START_WITH_INDEX}) | Window: {win} | File: {img})"
        else:
            msg = f"{n}/{m} (Index: {i} | Original Index: {i + START_WITH_INDEX}) | Window: {win})"
        
        print(msg)

        im = cv2.imread(img)
        cv2.imshow(win, im)
        
        next_imgs = check_dic[grp][i + 1 : i + NUM_WINDOWS]
        next_i = i + 1
        win_num += 1

        if NUM_WINDOWS > 1:
            cont_i = next_i
            for w, next_im in enumerate(next_imgs):
                next_win = f"Feature{win_num}"
                if VERBOSE:
                    pad_len = len(f"{n}/{m} ")
                    pad = " " * pad_len
                    msg_next = f"{pad}(Index: {cont_i} | Original Index: {next_i + START_WITH_INDEX + w}) | Window: {next_win} | File: {next_im})"
                    print(msg_next)
                next_im_data = cv2.imread(next_im)
                cv2.imshow(next_win, next_im_data)
                win_num += 1
                cont_i += 1
        
        #cv2.resizeWindow(next_win, 36, 76)
        if i == NUM_WINDOWS - 1:
            print("If all windows show a number, press any key to continue")
            cv2.waitKey(0)
        keypress = cv2.waitKey(WAIT_BEFORE_NEXT_IMG)
        if keypress == 27:
            # ESC key
            cv2.destroyAllWindows()
            sys.exit(0)
        if keypress == 32:
            # Space key
            print("Press any key to continue")
            cv2.waitKey(0)
            print("Press any key to continue")
        if keypress == 115:
            # s key
            NEW_DELAY = input("Enter new delay in ms between images\n")                
            
            ISINT = False
            while not ISINT:
                try:
                    WAIT_BEFORE_NEXT_IMG = int(NEW_DELAY)
                except ValueError as e:
                    print(f"Input was '{NEW_DELAY}' - not an INT!")
                    NEW_DELAY = input("Enter new delay in ms between images (must be an integer)\n")
                else:
                    ISINT = True
                    
                    
                
        n -= 1
        
    if END_WITH_INDEX or START_WITH_INDEX > 0:
        print("Press any key to quit\n")
        cv2.waitKey(0)
        #input("Press any key to quit\n")
    else:
        time.sleep(3)

cv2.destroyAllWindows()
