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

if os.path.isfile("data_sorted.pickle"):
    print("Opening data_sorted.pickle")
    with open('data_sorted.pickle', 'rb') as sorted:
         data = pickle.load(sorted)

#pp.pprint(data)
#{   'image_2021.03.18_09-15-13_cam2.jpg': {   0: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-0.png',
#                                                     '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-1.png'],
#                                              1: [],
#                                              2: [],
#                                              3: [],
#                                              4: [],
#                                              5: [],
#                                              6: [],
#                                              7: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-3.png'],
#                                              8: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-2.png'],
#                                              9: []},

fet_x_pos = {0: 20, 1: 111, 2: 204, 3: 301, 4: 395, 5: 486, 6: 590}
fet_y_pos = 250

digit = int(input("Please enter the digit you would like to see images for:\n"))

cv2.namedWindow("Original")
cv2.moveWindow("Original", 0,0)
cv2.namedWindow("Feature")
cv2.moveWindow("Feature", fet_x_pos[0],fet_y_pos)

#cv2.moveWindow("Feature", fet_x_pos[i],fet_y_pos)

for original in data.keys():
#    print(original)
    org_image = f"{TRAININGSET}/{original}"
    #print(org_image)
    org_image_data = cv2.imread(org_image)
    cv2.imshow("Original", org_image_data)
    #pp.pprint(data[original])

    for feat_image in data[original][digit]:
        #print(feat_image)
        featname = os.path.basename(feat_image)
        fet_image_data = cv2.imread(feat_image)
        cv2.imshow("Feature", fet_image_data)
        
        for key in fet_x_pos.keys():
            if featname == f"SingleDigitThreshhold-{key}.png":
                cv2.moveWindow("Feature", fet_x_pos[key],fet_y_pos)
        
        keypress = cv2.waitKey(1)
        if keypress == 27:
            sys.exit(0)
        elif keypress != 27 and keypress != -1:
            print(org_image)
            print(feat_image)
            time.sleep(5)

cv2.destroyAllWindows()
