#!/usr/bin/env python3.8
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
from termcolor import colored, cprint
from colorama import Fore, Back, Style

#./retrain_images/cold/2021.03.29/
#cold-2021.03.29_23-50-35-01-Original.png
#cold-2021.03.29_23-50-35-02-Original-BoundingBox.png
#cold-2021.03.29_23-50-35-03-Threshold.png
#cold-2021.03.29_23-50-35-04-ThresholdSingleDigit-3.png
#cold-2021.03.29_23-50-35-04-ThresholdSingleDigit-6.png
#cold-2021.03.29_23-50-35-05-ThresholdSingleDigit-3_reshaped.png
#cold-2021.03.29_23-50-35-05-ThresholdSingleDigit-6_reshaped.png
#cold-2021.03.29_23-56-06-01-Original.png
#cold-2021.03.29_23-56-06-02-Original-BoundingBox.png
#cold-2021.03.29_23-56-06-03-Threshold.png
#cold-2021.03.29_23-56-06-04-ThresholdSingleDigit-3.png
#cold-2021.03.29_23-56-06-05-ThresholdSingleDigit-3_reshaped.png

TRAININGSET = "./retrain_images"
DATAPATH_PICKLE = "data_path-retrain.pickle"
SORTED_PICKLE = "data_sorted-retrain.pickle"

timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
pp = pprint.PrettyPrinter(indent=4)

UNCHANGING_DIGITS = {
                    "cold": {
                                0: 0,
                                1: 1,
                                2: 0,
#                                3: 6,
#                                4: 8,
#                                5: 0,
#                                6: 0,
                            },
                    "warm": {
                                0: 0,
                                1: 0,
                                2: 5,
                                3: 1,
#                                4: 4,
#                                5: 0,
#                                6: 0,
                            },
                    }

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

def autosave(data, ts):
    autosavefile = bf = SORTED_PICKLE.replace(".pickle", f"_autosave-{ts}.pickle")
    
    cprint(f"------------------------------------------------------- Saving data to {autosavefile} --------------------------", "red")
    
    with open(autosavefile, "wb") as save:
        pickle.dump(data, save, pickle.HIGHEST_PROTOCOL)

def backup(data, ts):
    bf = SORTED_PICKLE.replace(".pickle", f"{ts}.pickle")
    cprint(f"------------------------------------------------------- Saving data to {bf}---------------------------------------------------------------------", "red")

    with open(bf, "wb") as save:
        pickle.dump(data, save, pickle.HIGHEST_PROTOCOL)

def save(data):
    savefile = SORTED_PICKLE
    cprint(f"------------------------------------------------------- Saving data to {savefile} -------------------------------------------------------", "red")
    
    with open(savefile, "wb") as save:
        pickle.dump(data, save, pickle.HIGHEST_PROTOCOL)

if os.path.isfile(DATAPATH_PICKLE):
    with open(DATAPATH_PICKLE, 'rb') as f:
        original = pickle.load(f)
else:

    firstlast = {}
    filepath = {}
    xfl = 0
    yfl = 0
    for woc in ["cold", "warm"]:
        filepath[woc] = sorted(glob.glob(f"{TRAININGSET}/{woc}/*/*-01-Original.png"))
        firstlast[woc] = [cv2.imread(filepath[woc][0]), cv2.imread(filepath[woc][-1])]

        firsti = 0
        lasti = len(filepath[woc])

        winname_first = f"{woc}{firsti}"
        winname_last = f"{woc}{lasti}"
        
        cv2.namedWindow(winname_first)
        cv2.resizeWindow(winname_first, 350, 100)
        cv2.moveWindow(winname_first, xfl, yfl)
        cv2.imshow(winname_first, firstlast[woc][0])

        
        cv2.namedWindow(winname_last)
        cv2.resizeWindow(winname_last, 350, 100)
        cv2.moveWindow(winname_last, xfl, yfl + 220)
        cv2.imshow(winname_last, firstlast[woc][1])
        
        xfl += 400

    cprint("Did you check the UNCHANGING_DIGITS dictionary?", "red")
    for i in UNCHANGING_DIGITS.keys():
        print(i)
        for j in UNCHANGING_DIGITS[i]:
            print(f"{Fore.CYAN}{j}{Style.RESET_ALL}: {Fore.YELLOW}{UNCHANGING_DIGITS[i][j]}{Style.RESET_ALL}")

    cprint("y/n\n", "red")
    keypress = cv2.waitKey(0)
    if keypress == 27:
        # ESC key
        cv2.destroyAllWindows()
        sys.exit(0)
    else:
        cv2.destroyAllWindows()

    if keypress != 121:
        cv2.destroyAllWindows()
        sys.exit(0)

    original = {}
    for woc in ["cold", "warm"]:
        original[woc] = {}
        for f in filepath[woc]:
            #fname = os.path.basename(f)
            fpath, fname = os.path.split(f)
            fid = fname.replace("-01-Original.png", "")
            original[woc][fname] = {"org": f}
            original[woc][fname]["feature"] = {}
            original[woc][fname]["feature_reshaped"] = {}
            original[woc][fname]["threshold"] = f"{fpath}/{fid}-03-Threshold.png"
            original[woc][fname]["boundingbox"] = f"{fpath}/{fid}-02-Original-BoundingBox.png"
            for i in range(0, 7, 1):
                featurefile = f"{fpath}/{fid}-04-ThresholdSingleDigit-{i}.png"
                if os.path.isfile(featurefile):
                    original[woc][fname]["feature"][i] = featurefile
                
                featurefile_reshaped = f"{fpath}/{fid}-05-ThresholdSingleDigit-{i}_reshaped.png"
                if os.path.isfile(featurefile_reshaped):
                    original[woc][fname]["feature_reshaped"][i] = featurefile_reshaped

    with open(DATAPATH_PICKLE, "wb") as f:
        #pp.pprint(original)
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(original, f, pickle.HIGHEST_PROTOCOL)

if os.path.isfile(SORTED_PICKLE):
    print(f"Opening {SORTED_PICKLE}")
    backupf = SORTED_PICKLE.replace(".pickle", f"-{timestamp}.pickle")
    shutil.copy2(SORTED_PICKLE, backupf)
    with open(SORTED_PICKLE, 'rb') as sorted:
         data = pickle.load(sorted)
else:
    data = {}

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

cv2.namedWindow("Original")
cv2.moveWindow("Original", 0,0)
cv2.namedWindow("BoundingBox")
cv2.moveWindow("BoundingBox", 0,155)
cv2.namedWindow("Threshold")
cv2.moveWindow("Threshold", 0,240)
cv2.namedWindow("Feature")
cv2.namedWindow("FeatureReshaped")

for i in range(0, 9, 1):
    win = f"Next{i}"
    cv2.namedWindow(win)

fet_x_pos = {0: 7, 1: 57, 2: 108, 3: 170, 4: 225, 5: 272, 6: 332}
fet_y_pos = 335

for woc in ["cold", "warm"]:
    for i in range(0, 7, 1):
        num_features = feature_count[woc][i]
        counter = 0
        
        original_list = list(original[woc].keys())
        
        for orig_i, f in enumerate(original_list):
            try:
                discard = data[f]
            except KeyError as e:
                data[f] = {}
                for grp in range(0, 10, 1):
                    data[f][grp] = []

            try:
                fet_image = original[woc][f]["feature"][i]
            except KeyError as e:
                cprint(f"Feature image for counter {i} not found - skipping", "red")
                continue
            try:
                fet_image_reshaped = original[woc][f]["feature_reshaped"][i]
            except KeyError as e:
                cprint(f"Reshaped feature image for counter {i} not found - skipping", "red")
                continue
            else:
                num_features -= 1
                processed = False
                for grp in data[f].keys():
                    if fet_image_reshaped in data[f][grp]:
                        #print(f"{fet_image} has already been processed")
                        processed = True
                        break

                if not processed:
                    cprint(f"Features remaining for position {i}: {num_features + 1}", "green")
                    #print("------------------------")

                    org_image = original[woc][f]["org"]
                    #print(f"{Style.DIM}{org_image}{Style.RESET_ALL}")
                    try:
                        org_image_data = cv2.imread(org_image)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Original image {org_image} not found - skipping!", "red")
                        continue
                    try:
                        cv2.imshow("Original", org_image_data)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Original image {org_image} could not be displayed - skipping!", "red")
                        continue


                    thr_image = original[woc][f]["threshold"]
                    #print(f"{Style.DIM}{thr_image}{Style.RESET_ALL}")
                    try:
                        thr_image_data = cv2.imread(thr_image)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Threshold image {thr_image} not found - skipping!", "red")
                        continue
                    try:
                        cv2.imshow("Threshold", thr_image_data)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Threshold image {thr_image} could not be displayed - skipping!", "red")
                        continue


                    bb_image = original[woc][f]["boundingbox"]
                    #print(f"{Style.DIM}{bb_image}{Style.RESET_ALL}")
                    try:
                        bb_image_data = cv2.imread(bb_image)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"BoundingBox image {bb_image} not found - skipping!", "red")
                        continue
                    try:
                        cv2.imshow("BoundingBox", bb_image_data)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"BoundingBox image {bb_image} could not be displayed - skipping!", "red")


                    #print(f"{Style.DIM}{fet_image}{Style.RESET_ALL}")
                    try:
                        fet_image_data = cv2.imread(fet_image)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Feature image {fet_image} not found - skipping!", "red")
                        continue
                    try:
                        cv2.imshow("Feature", fet_image_data)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Feature image {fet_image} could not be displayed - skipping!", "red")
                    else:
                        cv2.moveWindow("Feature", fet_x_pos[i],fet_y_pos)


                    #print(f"{Style.DIM}{fet_image_reshaped}{Style.RESET_ALL}")
                    try:
                        fet_image_reshaped_data = cv2.imread(fet_image_reshaped)
                        cv2.imshow("FeatureReshaped", fet_image_reshaped_data)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Reshaped feature image {fet_image_reshaped} not found - skipping!", "red")
                        continue
                    try:
                        cv2.imshow("FeatureReshaped", fet_image_reshaped_data)
                    except Exception as e:
                        #cprint(e, "yellow")
                        cprint(f"Reshaped feature image {fet_image_reshaped} could not be displayed - skipping!", "red")
                    else:
                        cv2.moveWindow("FeatureReshaped", fet_x_pos[i],fet_y_pos + 50)
                    
                    # ------------------------------------------------------------------------------------------------------------------------------
                    next_images = original_list[orig_i + 1 : orig_i + 11]
                    fet_y_pos_next = fet_y_pos + 150
                    for next_n, next_orig in enumerate(next_images):
                        win = f"Next{next_n}"
                        try:
                            f_next = original[woc][next_orig]["feature_reshaped"][i]
                        except (IndexError, KeyError)  as e:
                            pass
                        else:
                            fet_image_reshaped_next_data = cv2.imread(f_next)
                            cv2.imshow(win, fet_image_reshaped_next_data)
                            cv2.moveWindow(win, fet_x_pos[i], fet_y_pos_next)
                            fet_y_pos_next += 100
                            #cv2.waitKey(100)
                    # ------------------------------------------------------------------------------------------------------------------------------
                        
                    
                    if i in UNCHANGING_DIGITS[woc].keys():
                        digit = UNCHANGING_DIGITS[woc][i]
                        data[f][digit].append(fet_image_reshaped)
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
                            data[f][digit].append(fet_image_reshaped)
                        elif keypress == 98:
                            # b key
                            ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
                            backup(data, ts)
                            CHKINP = True
                            while CHKINP:
                                digit = int(input("Please provide digit:"))
                                if digit in range(0, 10, 1):
                                    print(digit)
                                    data[f][digit].append(fet_image_reshaped)
                                    CHKINP = False
                                    break
                                else:
                                    CHKINP = True
                        elif keypress == 113:
                            # s key
                            print("Saving and quitting")
                            save(data)
                            sys.exit(0)
                        else:
                            print(f"Unknow key {keypress}")
                            print("Image skipped")
                    #cprint(f"Features remaining for position {i}: {num_features}", "green")
                    if counter == 10:
                        ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
                        autosave(data, ts)
                        save(data)
                        counter = 0
                    else:
                        counter += 1
                        
#print('Data shape:', np.array(data[f]).shape)
ts = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
autosave(data, ts)
save(data)
cv2.destroyAllWindows()
cprint("Done! All imagees have been classified.", "green")
