#!/usr/bin/env python3
#
#

import os
import sys
import copy
import pprint
import pickle

#{   'image_2021.03.18_09-15-13_cam2.jpg': {   0: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-0.png',
#                                                     '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-1.png',
#                                                     '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-5.png'],
#                                              1: [],
#                                              2: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-4.png'],
#                                              3: [],
#                                              4: [],
#                                              5: [],
#                                              6: [],
#                                              7: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-3.png'],
#                                              8: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-2.png',
#                                                     '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-13_cam2.jpg_features/SingleDigitThreshhold-6.png'],
#                                              9: []},
#    'image_2021.03.18_09-15-18_cam2.jpg': {   0: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-18_cam2.jpg_features/SingleDigitThreshhold-0.png',
#                                                     '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-18_cam2.jpg_features/SingleDigitThreshhold-1.png',
#                                                     '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-18_cam2.jpg_features/SingleDigitThreshhold-5.png'],
#                                              1: [],
#                                              2: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-18_cam2.jpg_features/SingleDigitThreshhold-4.png'],
#                                              3: [],
#                                              4: [],
#                                              5: [],
#                                              6: [],
#                                              7: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-18_cam2.jpg_features/SingleDigitThreshhold-3.png'],
#                                              8: [   '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-18_cam2.jpg_features/SingleDigitThreshhold-2.png',
#                                                     '/home/blub/wasser/training-set/2021.03.18/cam2/image_2021.03.18_09-15-18_cam2.jpg_features/SingleDigitThreshhold-6.png'],
#                                              9: []},
#}

TRAINING_SET_OLD_PICKLE = "data_sorted.pickle"
TRAINING_SET_NEW_PICKLE = "data_sorted_new-path.pickle"
TRAININGSET_PATH_OLD = "/home/blub/wasser/training-set/2021.03.18/cam2"
TRAININGSET_PATH_NEW = "/home/blub/wasser/training/training-set/2021.03.18/cam2"

pp = pprint.PrettyPrinter(indent=4)

if os.path.isfile(TRAINING_SET_OLD_PICKLE):
    with open(TRAINING_SET_OLD_PICKLE, 'rb') as oldf:
        TRS = pickle.load(oldf)
else:
    print(f"File not found {TRAINING_SET_OLD_PICKLE}")
    sys.exit(0)

TRSOLD = copy.deepcopy(TRS)

for img in TRS.keys():
    for num in TRS[img]:
        changed = [p.replace(TRAININGSET_PATH_OLD, TRAININGSET_PATH_NEW) for p in TRS[img][num]]
        TRS[img][num] = changed

with open(TRAINING_SET_NEW_PICKLE, "wb") as newf:
        pickle.dump(TRS, newf, pickle.HIGHEST_PROTOCOL)

with open(TRAINING_SET_NEW_PICKLE, 'rb') as valf:
    TRSNEW = pickle.load(valf)

counter = 0

for img in TRSNEW.keys():
    print(f"{img} #######################################################################################")
    for num in TRSNEW[img]:
        print("---------------------------------")
        print("---- OLD")
        pp.pprint(TRSOLD[img][num])
        print("--- NEW")
        pp.pprint(TRSNEW[img][num])
        print("---------------------------------")
    counter +=1
    if counter > 1:
        break
