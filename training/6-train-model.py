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

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
from tensorflow.keras.callbacks import EarlyStopping



pp = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(description='Train the model')

parser.add_argument(
    '-i',
    '--input',
    dest='trainingdata_images',
    default='',
    type=str,
    help='Pickle file containing the images')

parser.add_argument(
    '-t',
    '--test',
    dest='use_n_for_test',
    default=10,
    type=int,
    help='Set number of images aside for testing the model')

parser.add_argument(
    '-n',
    '--same',
    dest='use_same_n',
    action='store_true',
    help='Use same count of images per class - default is to use all images available per class')

args = parser.parse_args()

#TRAININGDATA = "training-data-2021.03.30_19-58-48.pickle"
TRAININGDATA_IMAGES = args.trainingdata_images
USE_N_FOR_TEST = args.use_n_for_test * -1
USE_SAME_N = args.use_same_n

image_size = (76, 36)
img_rows, img_cols = image_size
input_shape = (img_rows, img_cols, 1)
num_classes = 10

batch_size = 500
epochs = 100
validation_split=0.2
shuffle=False

EARLYSTOP = False
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=10,
    min_delta=0.001, 
    mode='auto'
)

timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

if not TRAININGDATA_IMAGES:
    TRAININGDATA_IMAGES = sorted(glob.glob("image-set/*.pickle"))[-1]
    print(f"Use {Fore.CYAN}{TRAININGDATA_IMAGES}{Style.RESET_ALL} to train the model?")
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

# https://towardsdatascience.com/a-practical-introduction-to-early-stopping-in-machine-learning-550ac88bc8fd
def plot_metric(history, metric, f):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.savefig(f, format="png")
    #plt.show()

# Split up in training set and validation set
x_train = []
y_train = []
x_test = []
y_test = []

n_per_class = []
for i in range(0, 10, 1):
    n = len(digit_dict[i])
    print(f"{Fore.CYAN}{i: >3}{Style.RESET_ALL}: {Fore.YELLOW}{n}{Style.RESET_ALL}")
    n_per_class.append(n)

least_n = min(n_per_class)
max_n = max(n_per_class)
#print(n_per_class)
#print(least_n)
#print(max_n)

for i in range(0, 10, 1):
    shuffled_dict = digit_dict[i]
    random.shuffle(shuffled_dict)

    if USE_SAME_N:
        use_images = shuffled_dict[:least_n]
        print(f"{Fore.RED}Using {len(use_images)} images for class ${i}{Style.RESET_ALL}")
    else:
        use_images = shuffled_dict
        print(f"{Fore.RED}Using {len(use_images)} for class ${i}{Style.RESET_ALL}")

    xtrn = use_images[:USE_N_FOR_TEST]
    xtst = use_images[USE_N_FOR_TEST:]

    n = len(xtrn)
    x_train.extend(xtrn)
    y_train.extend([i] * n)
    
    n = len(xtst)
    x_test.extend(xtst)
    y_test.extend([i] * n)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#image_index = -1
#print(y_train[image_index])
#cv2.namedWindow("Feature")
#cv2.moveWindow("Feature", 0,0)
#cv2.imshow("Feature", x_train[image_index])
#cv2.waitKey(0)
#plt.imshow(x_train[image_index])
#plt.show()

# Preprocess the data (these are NumPy arrays)
# Model / data parameters

# the data, split between train and test sets

# Scale images to the [0, 1] range
#x_train = np.array(x_train).astype("float32") / 255
#x_test = np.array(x_test).astype("float32") / 255
# Make sure images have shape (28, 28, 1)
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)

print("Model shape:", input_shape)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        #keras.layers.InputLayer(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

model.compile(
                loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"]
)

if EARLYSTOP:
    CB = [early_stopping]
else:
    CB = []

history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            shuffle=shuffle,
            callbacks=CB
)

print("Evaluating model")
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save(f"model/wasser_model_{timestamp}.h5")

plot_metric(history, 'accuracy', f"model/wasser_model_{timestamp}_accuracy.png")
plot_metric(history, 'loss', f"model/wasser_model_{timestamp}_loss.png")
