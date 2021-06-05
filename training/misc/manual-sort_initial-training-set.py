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
import numpy as np
from datetime import datetime

#"training-set/2021.03.18/cam2/image_2021.03.18_11-11-12_cam2.jpg_features/SingleDigitThreshhold-0.png

TRAININGSET = "./training-set/2021.03.18/cam2"

timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
UNCHANGING_DIGITS = {0: 0, 1: 0, 2: 8}
cv2keydict = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8, 57: 9}

pp = pprint.PrettyPrinter(indent=4)

#{   'image_2021.03.18_09-15-13_cam2.jpg': {   'feature': {   0: '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-0.png',
#                                                             1: '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-1.png',
#                                                             2: '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-2.png',
#                                                             3: '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-3.png',
#                                                             4: '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-4.png',
#                                                             5: '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-5.png',
#                                                             6: '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-6.png'},
#                                              'org': '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg',
#                                              'threshold': '/home/pi/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg-1-threshhold.png'},

def autosave(data):
    autosavefile = "data_sorted.pickle"
    print(f"Saving data to {autosavefile}")
    
    with open(autosavefile, "wb") as autosave:
        pickle.dump(data, autosave, pickle.HIGHEST_PROTOCOL)

def backup(data, bf):
    print(f"Saving data to {bf}")

    with open(bf, "wb") as save:
        pickle.dump(data, save, pickle.HIGHEST_PROTOCOL)

#def save():
    

if os.path.isfile("data_path.pickle"):
    with open('data_path.pickle', 'rb') as f:
        original = pickle.load(f)
else:
    originalfilepath = sorted(glob.glob(f"{TRAININGSET}/image_*.jpg"))

    original = {}
    for f in originalfilepath:
        fname = os.path.basename(f)
        original[fname] = {"org": f}
        original[fname]["feature"] = {}
        original[fname]["threshold"] = f"{f}-1-threshhold.png"
        for i in range(0, 7, 1):
            featurefile = f"{TRAININGSET}/{fname}_features/SingleDigitThreshhold-{i}.png"
            if os.path.isfile(featurefile):
                original[fname]["feature"][i] = f"{TRAININGSET}/{fname}_features/SingleDigitThreshhold-{i}.png"

#    pp.pprint(original)

    with open("data_path.pickle", "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(original, f, pickle.HIGHEST_PROTOCOL)

if os.path.isfile("data_sorted.pickle"):
    print("Opening data_sorted.pickle")
    shutil.copy2("data_sorted.pickle", f"data_sorted-{timestamp}.pickle")
    with open('data_sorted.pickle', 'rb') as sorted:
         data = pickle.load(sorted)
else:
    data = {}

cv2.namedWindow("Original")
cv2.moveWindow("Original", 0,0)
cv2.namedWindow("Feature")

fet_x_pos = {0: 20, 1: 111, 2: 204, 3: 301, 4: 395, 5: 486, 6: 590}
fet_y_pos = 250

for i in range(0, 7, 1):
    counter = 0
    for f in original.keys():
        try:
            discard = data[f]
        except KeyError as e:
            data[f] = {}
            for grp in range(0, 10, 1):
                data[f][grp] = []

        try:
            fet_image = original[f]["feature"][i]
        except KeyError as e:
            print(f"Feature image for counter {i} not found - skipping")
            #cv2.namedWindow("Threshold")
            #cv2.moveWindow("Threshold", 0,340)
            #thr_image_data = cv2.imread(thr_image)
            #print(org_image)
            #print(thr_image)
            #cv2.imshow("Original", org_image_data)
            #cv2.imshow("Threshold", thr_image_data)
            #keypress = cv2.waitKey(0)
            #while keypress != 27:
            #    keypress = cv2.waitKey(0)
            #cv2.destroyWindow("Threshold")
        else:
            processed = False
            for grp in data[f].keys():
                if fet_image in data[f][grp]:
                    #print(f"{fet_image} has already been processed")
                    processed = True
                    break

            if not processed:
                org_image = original[f]["org"]
                org_image_data = cv2.imread(org_image)
                cv2.imshow("Original", org_image_data)
                thr_image = original[f]["threshold"]


                fet_image_data = cv2.imread(fet_image)
                cv2.imshow("Feature", fet_image_data)
                cv2.moveWindow("Feature", fet_x_pos[i],fet_y_pos)
                
                if i in UNCHANGING_DIGITS.keys():
                    digit = UNCHANGING_DIGITS[i]
                    data[f][digit].append(fet_image)
                    #print(digit)
                    #keypress = cv2.waitKey(1)
                    #if keypress == 27:
                    #    cv2.destroyAllWindows()
                    #    sys.exit(0)
                else:
                    #fet_image_data = cv2.imread(fet_image)
                    #cv2.imshow("Feature", fet_image_data)
                    #cv2.moveWindow("Feature", fet_x_pos[i],fet_y_pos)
                    keypress = cv2.waitKey(0)
                    if keypress == 27:
                        # ESC key
                        cv2.destroyAllWindows()
                        #autosave(data)
                        sys.exit(0)
                    elif keypress == 32:
                        # SPACE key
                        print("Image skipped")
                    elif keypress in cv2keydict.keys():
                        # 0-9 keys
                        digit = cv2keydict[keypress]
                        print(digit)
                        data[f][digit].append(fet_image)
                    elif keypress == 98:
                        ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
                        bf = f"data_sorted-{ts}.pickle"
                        backup(data, bf)
                        CHKINP = True
                        while CHKINP:
                            digit = int(input("Please provide digit:"))
                            if digit in range(0, 10, 1):
                                print(digit)
                                data[f][digit].append(fet_image)
                                CHKINP = False
                                break
                            else:
                                CHKINP = True
                    elif keypress == 113:
                        print("Saving and quitting")
                        autosave(data)
                        sys.exit(0)
                    else:
                        print(f"Unknow key {keypress}")
                        print("Image skipped")
                
                if counter == 10:
                    autosave(data)
                    counter = 0
                else:
                    counter += 1
                        
#print('Data shape:', np.array(data[f]).shape)
autosave(data)
cv2.destroyAllWindows()
