#!/home/pi/watermeter/bin/python3
#
#

import os
import sys
import cv2
import time
import shutil
import socket
import signal
import pprint
import datetime
import subprocess
import numpy as np
from gpiozero import LED
#from datetime import datetime
from influxdb import InfluxDBClient
from termcolor import colored, cprint
from colorama import Fore, Back, Style
from PIL import Image, ImageDraw, ImageFont, ImageColor

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

signal.signal(signal.SIGINT, signal.default_int_handler)

# In case your water meter has been swapped, the last value of the old meter can be added here
old_water_counter_value_warm = 0
old_water_counter_value_cold = 0

# Manually decide if a value should be written to InfluxDB or not
MANUAL = False
# If DRYRUN is set to True, no data will be written to InfluxDB
DRYRUN = False
# Be verbose!
DEBUG = True
# Display the croped image with bounding box, the single features and the recognized values?
VISUALIZE = True
# Define the display to show the captured images
os.environ['DISPLAY'] = ':0.0'
# Delete all saved images on start?
CLEAN_IMAGES_ON_START = False
# Save all unprocessed images as taken by the cam + intermediate and processed images
DEBUG_SAVE_IMAGESS = False
# Save images for which we faile to extract a feature
SAVE_ON_FEATURE_EXTRACT_FAILURE = False
# Save croped images and extracted features for for review into "test_dir"
# If SAVE_IMAGES_FOR_TEST is true, no images will be captured for re-training!
SAVE_IMAGES_FOR_TEST = False
# Save croped images and extracted features to train a model?
SAVE_FOR_TRAINING = False
# Save croped images and just the extracted features which scored below SAVE_IMAGES_BELOW_RETRAIN_THRESH during recognition phase
RETRAIN_THRESH = 0.99999
SAVE_IMAGES_BELOW_RETRAIN_THRESH = True

# Limit the increase from one to another recognition to avoid a false recogniced values make your water consumption plot go through the roof
# Flow (guideline)
#  Washstand - up to 6 l/min
#  Shower - up to 9 l/min
# 1 m³ = 1000 l
# 1 l  = 0.001 m³
#max_increase = {"warm": 0.05, "cold": 0.03}
max_increase_mm3 = {"warm": 0.020, "cold": 0.020}
#max_increase_mm3 = {"warm": 1.000, "cold": 1.000}

# Box position and size around all numbers to be extracted
big_box = {
                "warm": 
                    {
                        # Smaller number for X moves box more towards the right (end) of the numbers
                        "upper_left_x": 767,
                        # Smaller number for y moves box more towards the bottom of the numbers
                        "upper_left_y": 474,
                        "box_width": 348,
                        "box_height": 83,

                        # Threshold                        
                        "thresh_min": 200,
                        "thresh_max": 500,

                        # Single digit feature box size
                        "width_min": 10,
                        "width_max": 35,
                        "height_min": 33,
                        "height_max": 60,
                        
                        "rotate": cv2.ROTATE_180
                    },
                "cold":
                    {
                        # Smaller number for X moves box more towards the left (start) of the numbers
                        "upper_left_x": 877,
                        # Smaller number for Y moves box more towards the top of the numbers
                        "upper_left_y": 530,
                        "box_width": 351,
                        "box_height": 83,

                        # Threshold                        
                        "thresh_min": 160,
                        "thresh_max": 500,

                        # Single digit feature box size
                        "width_min": 10,
                        "width_max": 35,
                        "height_min": 33,
                        "height_max": 60,
                        
                        "rotate": False
                    }
                }

# Digits shown on the croped image
ndigits = 7
# How large is a section containing a number
section_size = {"warm": 50, "cold": 48}

# Video device to monitor warm and cold water meter
# Use udev rule to make this more consistent...
#camsetup = {
#            "warm": os.path.realpath("/dev/videoWARM"),
#            "cold": os.path.realpath("/dev/videoCOLD")
#            }

camsetup = {
            "warm": os.path.realpath("/dev/video2"),
            "cold": os.path.realpath("/dev/video0")
            }

# Configure to which pins of the RaspberryPi the LEDs are hooked up
ledsetup = {"warm": LED(17), "cold": LED(27)}
# Discrad the first n frames captured by the camera
discard_num_frames = 60
# Paus between taking photos of both warm and cold
interval = 30

# Setup dir structure to store images
img_dir = "/home/pi/watermeter/images"
retrain_dir = f"{img_dir}/retrain_images"
test_dir = f"{img_dir}/test_images"
error_img_dir = f"{img_dir}/error_images"
debug_img_dir = f"{img_dir}/debug_images"
feat_extract_error_folder = f"{img_dir}/extract_error_images"
image_dir_list = [retrain_dir, test_dir, error_img_dir, debug_img_dir, feat_extract_error_folder]

# Define the model which should be used for the image recognition
modelpath = "/home/pi/watermeter/model"
model_h5 = "wasser_model_2021.04.07_09-13-05.h5"
modelfile = f"{modelpath}/{model_h5}"

# Adjust feature image size according to what the model expects as input image size
image_size = (76, 36)
img_rows, img_cols = image_size

# Setup InfluxDB connection details
IFDB_IP = "127.0.0.1"
IFDB_PORT = 8086
IFDB_USER = "openhab"
IFDB_PW = "openhab"
IFDB_DB = "openhab"
IFDB_MEASUREMENT = "clepsydra"

ifdbc = InfluxDBClient(host=IFDB_IP,
                       port=IFDB_PORT,
                       username=IFDB_USER,
                       password=IFDB_PW,
                       database=IFDB_DB)

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 60)
pp = pprint.PrettyPrinter(indent=4)

def clean_dir_structure():
    # Since there probably will be a lot images and an "rm -f *" will exceed arguments:
    cprint(f"SETUPDS - Deleting content of {image_dir_list}.", "red")
    cprint("SETUPDS - Press ENTER to continue or anything else + ENTER to cancel file/folder deletion!", "red")
    response = input("")
    if response == "":
        for woc in camsetup.keys():
            for p in image_dir_list:
                p = f"{p}/{woc}"
                try:
                    filelist = os.listdir(p)
                except FileNotFoundError as e:
                    print(f"SETUPDS - Folder {p} does not exist - skipping")
                else:
                    for filename in filelist:
                        file_path = os.path.join(p, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                #print(f"SETUPDS - Deleting: {file_path}")
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                #print(f"SETUPDS - Deleting: {file_path}")
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f"SETUPDS - Failed to delete {file_path}")
                            print(e)
        cprint("SETUPDS - Done deleting all files and folders", "green")
    else:
        cprint("SETUPDS - Deletion of files and folders has been canceled!", "red")

def setup_dir_structure():
    for woc in camsetup.keys():
        for p in image_dir_list:
            p = f"{p}/{woc}"
            try:
                if DEBUG:
                    print(f"SETUPDS - Trying to create {p}")
                os.makedirs(p)
            except OSError as e:
                if DEBUG:
                    print(f"SETUPDS - Folder {p} already exist")
            else:
                if DEBUG:
                    print(f"SETUPDS - Folder {p} created")


def get_lock(process_name):
    get_lock._lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    try:
        get_lock._lock_socket.bind('\0' + process_name)
    except socket.error:
        print('lock exists, terminating')
        sys.exit()

get_lock('monitor-water')


def local_time_offset(t=None):
    """Return offset of local zone from GMT, either at present or at time t."""
    # python2.3 localtime() can't take None
    if t is None:
        t = time.time()

    if time.localtime(t).tm_isdst and time.daylight:
        return -time.altzone / 3600
    else:
        return -time.timezone / 3600


def write_to_ifdb(ifdbc, m3, currenttime, mytz, certainty):
    if old_water_counter_value_warm > 0:
        w = m3["warm"] + old_water_counter_value_warm
    else:
        w = m3["warm"]

    if old_water_counter_value_cold > 0:
        c = m3["cold"] + old_water_counter_value_cold
    else:
        c = m3["cold"]

    w_new = w
    c_new = c

    # Get the last measurement to make sure the recogniced value is actually higher or equal
    try:
        #ifresult = ifdbc.query(f'SELECT * FROM "{IFDB_MEASUREMENT}" GROUP BY * ORDER BY DESC LIMIT 1')
        ifresult = ifdbc.query(f'SELECT * FROM "{IFDB_MEASUREMENT}" WHERE ("Location" = \'Flat\' AND "Filtered" = \'Yes\') GROUP BY * ORDER BY DESC LIMIT 1')
    except Exception as e:
        cprint(f"INFLUXD - Error querying InfluxDB", "red")
        print(e)
    else:
        #[{'measurement': 'clepsydra', 'tags': 'Flat', 'fields': {'warm': 0.0, 'cold': 88.665}}]
        #print(ifresult)
        #ResultSet({'('clepsydra', {'Location': 'Flat'})': [{'time': '2021-03-21T14:05:27Z', 'cold': 88.684, 'warm': 44.881}]})
        m = list(ifresult.get_points(measurement=IFDB_MEASUREMENT))
        if DEBUG:
            cprint(f"INFLUXD - Last measurements in {IFDB_MEASUREMENT}:", "green")
            cprint(m, "green")
        #print('Result: ', m)
        #Result:  [{'time': '2021-03-21T14:05:27Z', 'cold': 88.684, 'warm': 44.881}]

        try:
            w_old = m[0]["warm"]
        except IndexError as e:
            print("INFLUXD - warm - Most likely your database is empty seeting w_old to 0")
            print(e)
            w_old = 0
            #increase_percent_w = 0
            increase_mm3_w = 0
        else:
            w_old = m[0]["warm"]
            
            if w < w_old:
                w = w_old

            try:
                #increase_percent_w = ((w - w_old) * 100) / w_old
                increase_mm3_w = w - w_old
            except ZeroDivisionError as e:
                w = w_old
                #increase_percent_w = 0
                increase_mm3_w = 0
            else:
                if MANUAL:
                    cprint(f"Write current warm value ({w} mm³) which differs by {increase_mm3_w:.3f} mm³ compared to last value ({w_old} mm³)?", "green")
                    write_yesno = input("y/n: ")
                    if write_yesno != "y":
                        w = w_old
                else:
                    #if increase_percent_w > max_increase["warm"]:
                    if increase_mm3_w > max_increase_mm3["warm"]:
                        if DEBUG:
                            #cprint(f"INFLUXD - warm - Current value ({w} mm³) increased {increase_percent_w}% compared to last value ({w_old} mm³) - ignoring!", "red")
                            cprint(f"INFLUXD - warm - Current value ({w} mm³) increased {increase_mm3_w:.3f} mm³ compared to last value ({w_old} mm³) - ignoring!", "red")
                        w = w_old

        try:
            c_old = m[0]["cold"]
        except IndexError as e:
            print("INFLUXD - cold - Most likely your database is empty seeting c_old to 0")
            print(e)
            c_old = 0
            #increase_percent_c = 0
            increase_mm3_c = 0
        else:
            c_old = m[0]["cold"]

            if c < c_old:
                c = c_old

            try:
                #increase_percent_c = ((c - c_old) * 100) / c_old
                increase_mm3_c = c - c_old
            except ZeroDivisionError as e:
                c = c_old
                #increase_percent_c = 0
                increase_mm3_c = 0
            else:
                if MANUAL:
                    cprint(f"Write current cold value ({c} mm³) which differs by {increase_mm3_c:.3f} compared to last value ({c_old} mm³)?", "green")
                    write_yesno = input("y/n: ")
                    if write_yesno != "y":
                        w = w_old
                else:
                    #if increase_percent_c > max_increase["cold"]:
                    if increase_mm3_c > max_increase_mm3["cold"]:
                        if DEBUG:
                            #cprint(f"INFLUXD - cold - Current value ({c} mm³) increased {increase_percent_c}% compared to last value ({c_old} mm³) - ignoring!", "red")
                            cprint(f"INFLUXD - cold - Current value ({c} mm³) increased {increase_mm3_c:.3f} mm³ compared to last value ({c_old} mm³) - ignoring!", "red")
                        c = c_old

        if DEBUG:
            cprint("INFLUXD - Last value from InfluxDB vs. current recognized value", "cyan")
            cprint(f"INFLUXD - warm - Last value from InfluxDB: {w_old} mm³", "yellow")
            #cprint(f"INFLUXD - warm - Current recognized value: {w_new} mm³ ({increase_percent_w}%)", "yellow")
            cprint(f"INFLUXD - warm - Current recognized value: {w_new} mm³ ({increase_mm3_w:.3f} mm³)", "yellow")
            cprint(f"INFLUXD - warm - Value decided to write  : {w} mm³", "yellow")
            cprint(f"INFLUXD - cold - Last value from InfluxDB: {c_old} mm³", "cyan")
            #cprint(f"INFLUXD - cold - Current recognized value: {c_new} mm³ ({increase_percent_c}%)", "cyan")
            cprint(f"INFLUXD - cold - Current recognized value: {c_new} mm³ ({increase_mm3_c:.3f} mm³)", "cyan")
            cprint(f"INFLUXD - cold - Value decided to write  : {c} mm³", "cyan")

        # Yes, I know clepsydra is actually a time measurement device that uses water - but hey, sounds cool, no? :)
        # 'time': datetime.date.strftime(currenttime, '%Y-%m-%dT%X.%z'),
        body_filtered = [
            {
            "measurement": IFDB_MEASUREMENT,
            "time": datetime.date.strftime(currenttime, '%Y-%m-%dT%X.%z'),
            "tags": {
                "Filtered": "Yes",
                "Location": "Flat"
                },
            "fields": {
                "warm": float(w),
                "cold": float(c)
                }
            }
        ]
        
        body_unfiltered = [
            {
            "measurement": IFDB_MEASUREMENT,
            "time": datetime.date.strftime(currenttime, '%Y-%m-%dT%X.%z'),
            "tags": {
                "Filtered": "No",
                "Location": "Flat"
                },
            "fields": {
                "warm": float(w_new),
                "cold": float(c_new)
                }
            }
        ]
        
        body_certainty = [
            {
            "measurement": IFDB_MEASUREMENT,
            "time": datetime.date.strftime(currenttime, '%Y-%m-%dT%X.%z'),
            "tags": {
                "Filtered": "No",
                "Location": "Flat",
                "Certainty": "warm",
                "Model": model_h5
                },
            "fields": certainty["warm"]
            },
            {
            "measurement": IFDB_MEASUREMENT,
            "time": datetime.date.strftime(currenttime, '%Y-%m-%dT%X.%z'),
            "tags": {
                "Filtered": "No",
                "Location": "Flat",
                "Certainty": "cold",
                "Model": model_h5
                },
            "fields": certainty["cold"]
            },
        ]
        
        body = body_filtered + body_unfiltered + body_certainty

        if DEBUG:
            print(f"INFLUXD - JSON body to be written to InfluxDB (DB: '{IFDB_DB}', Measurement: '{IFDB_MEASUREMENT}') - FILTERED")
            pp.pprint(body_filtered)
            print(f"INFLUXD - JSON body to be written to InfluxDB (DB: '{IFDB_DB}', Measurement: '{IFDB_MEASUREMENT}') - UNFILTERED")
            pp.pprint(body_unfiltered)
            print(f"INFLUXD - JSON body to be written to InfluxDB (DB: '{IFDB_DB}', Measurement: '{IFDB_MEASUREMENT}') - CERTAINTY")
            pp.pprint(body_certainty)
            print(f"INFLUXD - JSON body to be written to InfluxDB (DB: '{IFDB_DB}', Measurement: '{IFDB_MEASUREMENT}') - FILTERED + UNFILTERED + CERTAINTY")
            pp.pprint(body)
        
        if not DRYRUN:
            ifdbc.write_points(body)


def feature_extraction(frame, timestamp, currenttime, woc):
    if woc == "warm":
        #cap.set(cv2.CAP_PROP_FOCUS, 5)
        #cap.set(cv2.CAP_PROP_EXPOSURE, 554)
        #630x147,1008x1203
        #If we consider (0,0) as top left corner of image called im with left-to-right as x direction and top-to-bottom as y direction.
        #And we have (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region within that image, then:
        #roi = im[y1:y2, x1:x2]
        upper_left_x = big_box[woc]["upper_left_x"]
        upper_left_y = big_box[woc]["upper_left_y"]
        lower_right_x = upper_left_x + big_box[woc]["box_width"]
        lower_right_y = upper_left_y + big_box[woc]["box_height"]
        crop_size = [lower_right_x - upper_left_x, lower_right_y - upper_left_y]

        if DEBUG:
            print(f"FEATURE - {woc} - Crop size: {crop_size}")

        thresh_min = big_box[woc]["thresh_min"]
        thresh_max = big_box[woc]["thresh_max"]
        
        # Single digit feature box size
        width_min = big_box[woc]["width_min"]
        width_max = big_box[woc]["width_max"]
        height_min = big_box[woc]["height_min"]
        height_max = big_box[woc]["height_max"]
        
        imgc = frame[upper_left_y:lower_right_y, upper_left_x:lower_right_x]
        
        if big_box[woc]["rotate"]:
            imgc = cv2.rotate(imgc, big_box[woc]["rotate"])

        img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img.copy(), thresh_min, thresh_max, cv2.THRESH_BINARY_INV)

        # Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(img,(5,5),0)
        #ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        #for thresh_min in range(100, thresh_max, 5):
            #ret, thresh = cv2.threshold(img.copy(), thresh_min, thresh_max, cv2.THRESH_BINARY_INV)
            #cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-04-Crop-Gray-Threshold-{thresh_min}_{thresh_max}.png", thresh)

    if woc == "cold":
        #cap.set(cv2.CAP_PROP_FOCUS, 6)
        #cap.set(cv2.CAP_PROP_EXPOSURE, 554)
        #656x163,1122x1508
        #If we consider (0,0) as top left corner of image called im with left-to-right as x direction and top-to-bottom as y direction.
        #And we have (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region within that image, then:
        #roi = im[y1:y2, x1:x2]
        upper_left_x = big_box[woc]["upper_left_x"]
        upper_left_y = big_box[woc]["upper_left_y"]
        lower_right_x = upper_left_x + big_box[woc]["box_width"]
        lower_right_y = upper_left_y + big_box[woc]["box_height"]
        crop_size = [lower_right_x - upper_left_x, lower_right_y - upper_left_y]

        if DEBUG:
            print(f"FEATURE - {woc} - Crop size: {crop_size}")

        thresh_min = big_box[woc]["thresh_min"]
        thresh_max = big_box[woc]["thresh_max"]
        
        # Single digit feature box size
        width_min = big_box[woc]["width_min"]
        width_max = big_box[woc]["width_max"]
        height_min = big_box[woc]["height_min"]
        height_max = big_box[woc]["height_max"]

        imgc = frame[upper_left_y:lower_right_y, upper_left_x:lower_right_x]

        if big_box[woc]["rotate"]:
            imgc = cv2.rotate(imgc, big_box[woc]["rotate"])

        img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img.copy(), thresh_min, thresh_max, cv2.THRESH_BINARY_INV)

        # Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(img,(5,5),0)
        #ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        #for thresh_min in range(100, thresh_max, 5):
            #ret, thresh = cv2.threshold(img.copy(), thresh_min, thresh_max, cv2.THRESH_BINARY_INV)
            #cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-04-Crop-Gray-Threshold-{thresh_min}_{thresh_max}.png", thresh)

    if DEBUG_SAVE_IMAGESS:
        cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-00-Frame.png", frame)
        cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-01-Original.png", imgc)
        cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-01-Original-Crop-Gray.png", img)
        cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-03-Threshold.png", thresh)

    # find contours in the thresholded image, then initialize the digit contours lists
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    found_cnts = len(cnts)
    if DEBUG:
        print(f"FEATURE - {woc} - Number of contours is: {found_cnts}")
    


    img_bb = imgc.copy()
    img_th = thresh.copy()
    digitCnts = []
    digitBdBx = []
    
    for i, c in enumerate(cnts):
        # compute the bounding box of the contour
        area = cv2.contourArea(c)
        br = cv2.boundingRect(c)
        x, y, w, h = br

        if DEBUG:
            print(f"FEATURE - {woc} - Bounding box {i}: {x},{y} | {w},{h}")

        if (w >= width_min and w <= width_max) and (h >= height_min and h <= height_max):
            if DEBUG:
                print(f"FEATURE - {woc} - Bounding box {i} matches size criteria ({w} >= {width_min} and {w} <= {width_max}) and ({h} >= {height_min} and {h} <= {height_max})")
            digitCnts.append(c)
            digitBdBx.append(br)
            # Draw small green contour line around all found countours in the original image which match our selection
            cv2.drawContours(img_bb, c, -1, (0, 255, 0, 255), 1)
            cv2.drawContours(img_th, c, -1, (0, 255, 0, 255), 1)
            # Draw small green box around all found contours in the original image which match our selection
            cv2.rectangle(img_bb,(x,y),(x+w,y+h),(0, 255, 0, 255),1)
            cv2.rectangle(img_th,(x,y),(x+w,y+h),(0, 255, 0, 255),1)
            # Add a red number to the drawn box for identification purpose
            #cv2.putText(img, text, position, fontFace, fontScale, color (BGR), thickness, lineType, bottomLeftOrigin)
            cv2.putText(img_bb, str(i), (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255, 255), 1)
            cv2.putText(img_th, str(i), (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255, 255), 1)
        else:
            # Draw small blue contour line around all found countours in the original image
            cv2.drawContours(img_bb, c, -1, (255, 0, 0, 255), 1)
            cv2.drawContours(img_th, c, -1, (255, 0, 0, 255), 1)
            # Draw small blue box around all found contours in the original image
            cv2.rectangle(img_bb, (x,y), (x+w,y+h), (255, 0, 0, 255), 1)
            cv2.rectangle(img_th, (x,y), (x+w,y+h), (255, 0, 0, 255), 1)
            # Add a yellow number to the drawn box for identification purpose
            #cv2.putText(img, text, position, fontFace, fontScale, color (BGR), thickness, lineType, bottomLeftOrigin)
            cv2.putText(img_bb, str(i), (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (3, 233, 252, 255), 1)
            cv2.putText(img_th, str(i), (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (3, 233, 252, 255), 1)
        
    if DEBUG:
        print(f"FEATURE - {woc} - Writing images with contours and bounding boxes")
    if DEBUG_SAVE_IMAGESS:
        cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-05-BoundingBox-Org.png", img_bb)
        cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-06-BoundingBox-Thres.png", img_th)
    
    # sort the contours from left-to-right
    # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
    features = {"original": {}, "resized": {}, "reshaped": {}}

    try:
        (digitCnts, digitBdBx) = zip(*sorted(zip(digitCnts, digitBdBx), key=lambda b:b[1][0], reverse=False))
    except ValueError as e:
        print(f"FEATURE - {woc} - Error sorting contours from left to right")
        print(e)
        cv2.imwrite(f"{error_img_dir}/{woc}/{woc}-{timestamp}-05-BoundingBox-Org.png", img_bb)
        cv2.imwrite(f"{error_img_dir}/{woc}/{woc}-{timestamp}-06-BoundingBox-Thres.png", img_th)
    else:
        digits = []
        digits_contour = []

        current_day = datetime.datetime.now().strftime("%Y.%m.%d")
        
        if SAVE_IMAGES_FOR_TEST:
            day_test_folder = f"{test_dir}/{woc}/{current_day}"
            
            try:
                os.makedirs(day_test_folder)
            except OSError as e:
                pass

            cv2.imwrite(f"{day_test_folder}/{woc}-{timestamp}-01-Original.png", imgc)
            cv2.imwrite(f"{day_test_folder}/{woc}-{timestamp}-02-Original-BoundingBox.png", img_bb)
            cv2.imwrite(f"{day_test_folder}/{woc}-{timestamp}-03-Threshold.png", thresh)

        if SAVE_FOR_TRAINING and not SAVE_IMAGES_FOR_TEST:
            if DEBUG:
                cprint(f"FEATURE - {woc} - Saving all features for training purpose", "red")

            day_retrain_folder = f"{retrain_dir}/{woc}/{current_day}"

            try:
                os.makedirs(day_retrain_folder)
            except OSError as e:
                pass

            cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-01-Original.png", imgc)
            cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-02-Original-BoundingBox.png", img_bb)
            cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-03-Threshold.png", thresh)

        found_valid_cnts = len(digitCnts)

        if DEBUG:
            if found_valid_cnts < ndigits:
                cprint(f"FEATURE - {woc} - Found {found_valid_cnts}/{ndigits} digits", "red")
            else:
                cprint(f"FEATURE - {woc} - Found {found_valid_cnts}/{ndigits} digits", "green")

        if SAVE_ON_FEATURE_EXTRACT_FAILURE:
            if found_valid_cnts < ndigits:
                cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-00-Frame.png", frame)
                cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-01-Original.png", imgc)
                cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-01-Original-Crop-Gray.png", img)
                cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-02-Original-BoundingBox.png", img_bb)
                cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-03-Threshold.png", thresh)

        for i, (c, b) in enumerate(zip(digitCnts, digitBdBx)):
            x, y, w, h = b
            #print(f"Size: {x}, {y}, {w}, {h}")
            #print("---------------------------------")
            
            # Assign features to counter positions where 0 is most left - in case no threshold could be found
            secsize = crop_size[0] / ndigits
            #secsize = section_size[woc]

            if DEBUG:
                print(f"FEATURE - {woc} - Section size: {secsize}")

            secsize_start = 0
            secsize_stop = secsize
            pos = ndigits - 1
            locked_pos = False
            for position in range(0, ndigits, 1):
                if x > secsize_start and x <= secsize_stop:
                    pos = position
                    locked_pos = True
                    if DEBUG:
                        cprint(f"FEATURE - {woc} - Position {position}: {x} > {secsize_start} and {x} <= {secsize_stop}", "green")
                    break
                else:
                    if DEBUG:
                        cprint(f"FEATURE - {woc} - Position {position}: {x} > {secsize_start} and {x} <= {secsize_stop}", "cyan")
                    secsize_start += secsize
                    secsize_stop += secsize
            
            if DEBUG:
                if locked_pos:
                    cprint(f"FEATURE - {woc} - Got lock on Position {pos}!", "green")
                else:
                    cprint(f"FEATURE - {woc} - No lock on Position!", "red")


#            if x <= secsize * 1:
#                pos = 0
#            if x > secsize * 1 and x <= secsize * 2:
#                pos = 1
#            if x > secsize * 2 and x <= secsize * 3:
#                pos = 2
#            if x > secsize * 3 and x <= secsize * 4:
#                pos = 3
#            if x > secsize * 4 and x <= secsize * 5:
#                pos = 4
#            if x > secsize * 5 and x <= secsize * 6:
#                pos = 5
#            if x > secsize * 6:
#                pos = 6

            # Create a single threshhold image for each digit
            digittresh = thresh[y:y + h, x:x + w]
            features["original"][pos] = digittresh
            if DEBUG_SAVE_IMAGESS:
                cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-07-SingleDigitThreshhold-{pos}.png", digittresh)

            digittresh_resized = cv2.resize(digittresh, (img_cols, img_rows))
            features["resized"][pos] = digittresh_resized
            if DEBUG_SAVE_IMAGESS:
                cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-08-SingleDigitThreshhold-{pos}_resized.png", digittresh_resized)

            digittresh_reshaped = digittresh_resized.reshape(img_rows, img_cols, 1).astype(np.float32)
            features["reshaped"][pos] = digittresh_reshaped
            
            if DEBUG_SAVE_IMAGESS:
                cv2.imwrite(f"{debug_img_dir}/{woc}/{woc}-{timestamp}-09-SingleDigitThreshhold-{pos}_resized_reshaped.png", digittresh_reshaped)
                
            if SAVE_IMAGES_FOR_TEST:
                cv2.imwrite(f"{day_test_folder}/{woc}-{timestamp}-04-ThresholdSingleDigit-{pos}.png", features["original"][pos])
                cv2.imwrite(f"{day_test_folder}/{woc}-{timestamp}-05-ThresholdSingleDigit-{pos}_reshaped.png", features["reshaped"][pos])

            if SAVE_FOR_TRAINING and not SAVE_IMAGES_FOR_TEST:
                cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-04-ThresholdSingleDigit-{pos}.png", features["original"][pos])
                cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-05-ThresholdSingleDigit-{pos}_reshaped.png", features["reshaped"][pos])
            
            if found_valid_cnts < ndigits:
                if SAVE_ON_FEATURE_EXTRACT_FAILURE:
                    #cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-00-Frame.png", imgc)
                    #cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-01-Original.png", imgc)
                    #cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-01-Original-Crop-Gray.png", img)
                    #cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-02-Original-BoundingBox.png", img_bb)
                    #cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-03-Threshold.png", thresh)
                    cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-04-ThresholdSingleDigit-{pos}.png", features["original"][pos])
                    cv2.imwrite(f"{feat_extract_error_folder}/{woc}/{woc}-{timestamp}-05-ThresholdSingleDigit-{pos}_reshaped.png", features["reshaped"][pos])

    return features, thresh, imgc, img_bb

def create_img(best):
    if best == "X":
        color = "red"
        best = 0
    else:
        color = "green"

    rec_img_value_pil = Image.new('RGB', (image_size[1],image_size[0]), color = 'white')
    d = ImageDraw.Draw(rec_img_value_pil)
    d.text((0,0), f"{best}", font=font, fill=color)
    rec_img_value_cv = cv2.cvtColor(np.array(rec_img_value_pil), cv2.COLOR_RGB2BGR)

    return rec_img_value_cv
        

def recognition(features, thresh, imgc, img_bb, woc):
    #numbers = {0: "X", 1: "X", 2: "X", 3: "X", 4: "X", 5: "X", 6: "X"}
    numbers = {}
    certainty = {}
    
    if VISUALIZE:
        cv2.imshow(visu_dict[woc]["bbox"], img_bb)
        cv2.imshow(visu_dict[woc]["thresh"], thresh)

    for pos in range(0, ndigits, 1):
        try:
            feat = features["reshaped"][pos]
        except KeyError as e:
            numbers[pos] = "0"
            certainty[int(pos)] = float(-1)
            if VISUALIZE:
                cv2.imshow(visu_dict[woc]["Rec"][pos], create_img("X"))
        else:
            predictlist = []
            predictlist.append(feat)
            predictlist = np.array(predictlist)
            
            if DEBUG:
                print(f"RECOGNI - {woc} - Model and input shape information ---")
                print(f"RECOGNI - {woc} - Model expects input shape (1, 76, 36, 1)")
                print(f"RECOGNI - {woc} - Input shape:", predictlist.shape)
                print(predictlist.shape[0], "input samples")
            
            # predict digit
            predictions = model.predict(predictlist)
            best = predictions.argmax()
            accuracy = predictions[0][best]
            percent = accuracy * 100
            if DEBUG:
                cprint(f"RECOGNI - {woc} - Best prediction for type at position {pos}: {best} ({percent:7.3f}%)", "cyan")
                #print(predictions.tostring())
                ps = ' | '.join([str(e) for e in predictions[0]])
                print(ps)

            # Ugly temp fix/hack until model gets better - Set first digit to 0 as it will most likely never change before the counter has to be replaced:
            #----------------------------------------------------------------------------------------------------------------------------------------
            #if pos <= 0 and woc == "warm":
            #    numbers[pos] = 0
            #else:
            #    numbers[pos] = int(best)

            # Ugly temp fix/hack until model gets better - Set first digit to 0 as it will most likely never change before the counter has to be replaced:
            #-----------------------------------------------------------------------------------------------------------------------------------
            #if pos <= 0 and woc == "cold":
            #    numbers[pos] = 0
            #else:
            #    numbers[pos] = int(best)
            
            numbers[pos] = int(best)
            
            certainty[int(pos)] = float(predictions[0][best])
            
            if predictions[0][best] < RETRAIN_THRESH and SAVE_FOR_TRAINING:
                cprint(f"RECOGNI - {woc} - Best prediction for type at position {pos}: {best} is below {RETRAIN_THRESH} ({predictions[0][best]})", "yellow")
    
            if predictions[0][best] < RETRAIN_THRESH and SAVE_IMAGES_BELOW_RETRAIN_THRESH and not SAVE_FOR_TRAINING:
                cprint(f"RECOGNI - {woc} - Best prediction for type at position {pos}: {best} is below {RETRAIN_THRESH} ({predictions[0][best]}) - Saving images for re-training", "yellow")

                current_day = datetime.datetime.now().strftime("%Y.%m.%d")
                day_retrain_folder = f"{retrain_dir}/{woc}/{current_day}"

                if not SAVE_IMAGES_FOR_TEST and not SAVE_FOR_TRAINING:
                    try:
                        os.makedirs(day_retrain_folder)
                    except OSError as e:
                        pass
                    
                    cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-01-Original.png", imgc)
                    cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-02-Original-BoundingBox.png", img_bb)
                    cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-03-Threshold.png", thresh)
                    cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-04-ThresholdSingleDigit-{pos}.png", features["original"][pos])
                    cv2.imwrite(f"{day_retrain_folder}/{woc}-{timestamp}-05-ThresholdSingleDigit-{pos}_reshaped.png", features["reshaped"][pos])
            
            if VISUALIZE:
                cv2.imshow(visu_dict[woc]["Feature"][pos], features["reshaped"][pos])
                cv2.imshow(visu_dict[woc]["Rec"][pos], create_img(best))
    
    if DEBUG:
        print(f"RECOGNI - {woc} - Position-Value dict: {numbers}")

    tmplist = []
    for i in range(0, ndigits, 1):
        tmplist.append(str(numbers[i]))

    m3_tmp = "".join(tmplist)
    try:
        m3_tmp = int(m3_tmp)
    except ValueError as e:
        if DEBUG:
            cprint(f"RECOGNI - {woc} - Recognized value {m3_tmp} is not a number!", "red")
        m3 = 0
        pass
    else:
        m3 = float(m3_tmp/1000)
    
    cprint(f"RECOGNI - {woc} - Value for m3 recognized: {m3}", "green")
    return m3, certainty


def get_image_opencv(woc, timestamp):
    ledsetup[woc].on()
    cv2dev = int(camsetup[woc].replace("/dev/video", ""))
    cap = cv2.VideoCapture(cv2dev)
    
    AUTOFOCUS = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    #print("AutoFocus: ", AUTOFOCUS)
    
    AUTOEXP = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    #print("AutoExp: ", AUTOEXP)
    
    FOCUS = cap.get(cv2.CAP_PROP_FOCUS)
    #print("Focus: ", FOCUS)

    EXP = cap.get(cv2.CAP_PROP_EXPOSURE)
    #print("Exp: ", EXP)
    
    # Change the camera setting using the set() function
    #cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    for i in range(1, discard_num_frames + 1, 1):
        ret, frame = cap.read()

    cap.release()
    ledsetup[woc].off()

    return frame


if __name__ == "__main__":
    if CLEAN_IMAGES_ON_START:
        st = time.time()
        clean_dir_structure()
        et = time.time()
        if DEBUG:
            print(f"MAINPRG - Runtime clean_dir_structure: {et - st}")

    st = time.time()
    setup_dir_structure()
    et = time.time()

    if DEBUG:
        print(f"MAINPRG - Runtime setup_dir_structure: {et - st}")
        
    if DEBUG:
        print(f"MAINPRG - Loading model {modelfile}")
    # Model expects input shape (1, 76, 36, 1)
    model = load_model(modelfile)
    if DEBUG:
        model.summary()

    if VISUALIZE:
        visu_dict = {}

        for i, woc in enumerate(camsetup.keys()):
            space_between_warm_and_cold = 10 * i
            visu_dict[woc] = {}
            visu_dict[woc]["Feature"] = {}
            visu_dict[woc]["Rec"] = {}
            win_decoration_size = 50

            # Windows to display the croped image with bounding boxes
            crop_x_pos = 0 + (big_box[woc]["box_width"] * i) + space_between_warm_and_cold
            crop_y_pos = 100
            winname_vis_bbox = f"{woc} - Crop with bounding boxes"
            visu_dict[woc]["bbox"] = winname_vis_bbox
            cv2.namedWindow(winname_vis_bbox)
            cv2.resizeWindow(winname_vis_bbox, big_box[woc]["box_width"], big_box[woc]["box_height"])
            cv2.moveWindow(winname_vis_bbox,  crop_x_pos, crop_y_pos)
            
            # Windows to display the croped threshold image
            thresh_x_pos = crop_x_pos
            thresh_y_pos = crop_y_pos + big_box[woc]["box_height"] + win_decoration_size
            winname_vis_thresh = f"{woc} - Crop threshold"
            visu_dict[woc]["thresh"] = winname_vis_thresh
            cv2.namedWindow(winname_vis_thresh)
            cv2.resizeWindow(winname_vis_thresh, big_box[woc]["box_width"], big_box[woc]["box_height"])
            cv2.moveWindow(winname_vis_thresh,  thresh_x_pos, thresh_y_pos)
            
            # Windows to display the extracted single digit features
            
            for j in range(0, ndigits, 1):
                
                f_hight = image_size[0]
                f_width = image_size[1]
                
                feat_x_pos = crop_x_pos + (section_size[woc] * j) + space_between_warm_and_cold
                feat_y_pos = thresh_y_pos + big_box[woc]["box_height"] + win_decoration_size

                winname_vis_feat = f"{woc} - Feature pos {j}"
                visu_dict[woc]["Feature"][j] = winname_vis_feat
                cv2.namedWindow(winname_vis_feat)
                cv2.moveWindow(winname_vis_feat, feat_x_pos, feat_y_pos)
                cv2.resizeWindow(winname_vis_feat, f_width, f_hight)
                
                winname_vis_rec = f"{woc} - Recognized pos {j}"
                visu_dict[woc]["Rec"][j] = winname_vis_rec
                cv2.namedWindow(winname_vis_rec)
                cv2.moveWindow(winname_vis_rec, feat_x_pos, feat_y_pos + f_hight + win_decoration_size)
                cv2.resizeWindow(winname_vis_rec, f_width, f_hight)
            
    while True:
        try:
            timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
            mytz = datetime.timezone(datetime.timedelta(hours=local_time_offset()))
            currenttime = datetime.datetime.now(tz=mytz)

            cbm = {}
            certain = {}
            for i, woc in enumerate(camsetup.keys()):
                st = time.time()
                frame = get_image_opencv(woc, timestamp)
                et = time.time()
                if DEBUG:
                    print(f"MAINPRG - Runtime get_image: {et - st}")
                
                st = time.time()
                #feature_extraction_v2(frame, timestamp, currenttime, woc)
                features, thresh, imgc, img_bb = feature_extraction(frame, timestamp, currenttime, woc)
                et = time.time()
                if DEBUG:
                    print(f"MAINPRG - Runtime feature_extraction: {et - st}")
                
                st = time.time()
                cbm[woc], certain[woc] = recognition(features, thresh, imgc, img_bb, woc)
                et = time.time()

                if DEBUG:
                    print(f"MAINPRG - Runtime recognition: {et - st}")

                if VISUALIZE:
                    cv2.waitKey(500)
            
            st = time.time()
            write_to_ifdb(ifdbc, cbm, currenttime, mytz, certain)
            et = time.time()
            if DEBUG:
                print(f"MAINPRG - Runtime write_to_ifdb: {et - st}")
        
            if DEBUG:
                t = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
                print(f"MAINPRG - {t}")
                print(f"MAINPRG - {Fore.GREEN}Sleeping for {interval} seconds{Style.RESET_ALL}")
            time.sleep(interval)

        except KeyboardInterrupt as e:
            print("Terminating...")
            print(e)
            for woc in camsetup.keys():
                ledsetup[woc].off()
            cv2.destroyAllWindows()
            sys.exit(0)

#        except cv2.error as e:
#            cprint("OpenCV error - sleeping for {interval} sec", "red")
#            cprint(e, "yellow")
#            time.sleep(interval)
            
        except Exception as e:
            #cprint(f"Something unexpected went wrong - sleeping for {interval} sec before re-starting the script", "red")
            cprint(f"Something unexpected went wrong - quitting", "red")
            cprint(e, "red")
            sys.exit(1)
            #try:
            #    for woc in camsetup.keys():
            #        ledsetup[woc].off()
            #    cv2.destroyAllWindows()
            #    ifdbc.close()
            #except Exception as e:
            #    pass
            #time.sleep(interval)
            #os.execv(__file__, sys.argv)
