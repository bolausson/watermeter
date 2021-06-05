#!/usr/bin/env python3
#
#

import os
import sys
import cv2
import glob
import h5py
import time
import pickle
import pprint
import socket
import signal
import random
import logging
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored, cprint
from colorama import Fore, Back, Style

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

pp = pprint.PrettyPrinter(indent=4)

image_size = (76, 36)
img_rows, img_cols = image_size
input_shape = (img_rows, img_cols, 1)

print("Use space key for next image")
print("Use Esc key to exit")

#modelfiles = ["wasser_model_2021.03.19_16-45-11.h5", "wasser_model_2021.03.30_23-43-25.h5", "wasser_model_2021.04.01_22-34-19.h5", "wasser_model_2021.04.01_22-46-59.h5"]
modelfiles = sorted(glob.glob("model/wasser_model_*.h5"))
TESTINGDATA_IMAGES = "test_images/test_image-set.pickle"
#filelist = sorted(glob.glob("test_images/*/2021.*/*_reshaped.png"))

with open(TESTINGDATA_IMAGES, 'rb') as f:
    digit_dict = pickle.load(f)

model = {}
model_stats = {}
scoredict = {}

for i, modelfile in enumerate(modelfiles):
    scoredict[modelfile] = {"Overall": {"Hit": 0, "Miss": 0, "Tests": 0}}
    for key in digit_dict.keys():
        scoredict[modelfile][key] = {"Hit": 0, "Miss": 0, "Tests": 0}
    p, f = os.path.split(modelfile)
    print(f"Loading model {f} ({modelfile})")
    m = load_model(modelfile)
    m.summary()
    model[modelfile] = m
    model_stats[modelfile] = 0

#random.shuffle(filelist)

def test_models():
    for key in digit_dict.keys():
        numfeat = len(digit_dict[key])
        for i, imgfile in enumerate(digit_dict[key]):
            #winname = f"Feature-{i}"
            winname = "Feature"
            cv2.namedWindow(winname)
            #cv2.moveWindow(winname, 0, (i + 1) * 80)

            imglist = []
            
            #img = tf.keras.preprocessing.image.load_img(imgfile, color_mode="grayscale", target_size=image_size)
            #imga = keras.preprocessing.image.img_to_array(img)

            imglist.append(imgfile)
            imglist = np.array(imglist)
            
            #print("Input shape:", imglist.shape)
            #print(imglist.shape[0], "input samples")

            # predict digit
            #print(f"--------------------- Predicting feature {i}/{numfeat} from bucket {key}---------------------")
            #cv2.imshow(winname, imgfile)
            for j, m in enumerate(model.keys()):
                predictions = model[m].predict(imglist)
                #print(predictions)
                best = predictions.argmax()
                accuracy = predictions[0][best]
                percent = accuracy * 100

                if accuracy == 1:
                    c = Fore.GREEN
                    #print(f"Best match: {c}{best}{Style.RESET_ALL} with {c}{accuracy:0.16f} accuracy{Style.RESET_ALL} using model {c}{m}{Style.RESET_ALL}")
                    keypress = cv2.waitKey(1)
                else:
                    c = Fore.YELLOW
                    print(f"Matched {key} as {c}{best}{Style.RESET_ALL} with {c}{accuracy:0.16f} accuracy{Style.RESET_ALL} using model {c}{m}{Style.RESET_ALL}")
                    cv2.imshow(winname, imgfile)
                    keypress = cv2.waitKey(1)
                    
                if best == key:
                    scoredict[m][key]["Hit"] += 1
                    scoredict[m][key]["Tests"] += 1
                    scoredict[m]["Overall"]["Hit"] += 1
                    scoredict[m]["Overall"]["Tests"] += 1
                else:
                    print(f"{Fore.RED}{key} INCORRECT recognized{Style.RESET_ALL} as {c}{best}{Style.RESET_ALL} with {c}{accuracy:0.16f} accuracy{Style.RESET_ALL} using model {c}{m}{Style.RESET_ALL}")
                    scoredict[m][key]["Miss"] += 1
                    scoredict[m][key]["Tests"] += 1
                    scoredict[m]["Overall"]["Miss"] += 1
                    scoredict[m]["Overall"]["Tests"] += 1
                model_stats[m] += accuracy
            
                if keypress == 27:
                    # ESC key
                    cv2.destroyAllWindows()
                    return

#    max_score = i + 1
#    print(f'\nMax possible "score": {max_score}')


#    model_sorted = sorted(model_stats, key=model_stats.get, reverse=True)
#    print(f"\n{Fore.CYAN}Sorted by model with higest sum of recognition probability{Style.RESET_ALL}")

#    for i, item in enumerate(model_sorted):
#        if i == 0:
#            winner = item
#            print(f"{Fore.GREEN}{item}{Style.RESET_ALL}: {model_stats[item]}")
#        else:
#            print(f"{item}: {model_stats[item]}")

#    print(f"\n{Fore.CYAN}Sorted by model name{Style.RESET_ALL}")
#    #pp.pprint(model_stats)
#    #model_stats_sorted = sorted(model_stats, key=model_stats.__getitem__)
#    for item in sorted(model_stats):
#        if item == winner:
#            print(f"{Fore.GREEN}{item}{Style.RESET_ALL}: {model_stats[item]}")
#        else:
#            print(f"{item}: {model_stats[item]}")
#
#    sys.exit(0)

test_models()
cv2.destroyAllWindows
pp.pprint(scoredict)
