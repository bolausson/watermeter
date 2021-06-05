#!/usr/bin/env python3
#
#

import os
import sys
import cv2
import h5py
import time
import socket
import signal
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from gpiozero import LED

signal.signal(signal.SIGINT, signal.default_int_handler)

img_dir = "/home/pi/wasser/images"

def get_lock(process_name):
    # Without holding a reference to our socket somewhere it gets garbage
    # collected when the function exits
    get_lock._lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    try:
        # The null byte (\0) means the socket is created 
        # in the abstract namespace instead of being created 
        # on the file system itself.
        # Works only in Linux
        get_lock._lock_socket.bind('\0' + process_name)
        #print('I got the lock')
    except socket.error:
        print('lock exists, terminating')
        sys.exit()

get_lock('create-dataset-fswebcam')

def detect_edge(img_name, imgfullpath, feature_path, width_min=25, width_max=60, height_min=60, height_max=110):
    #print(f"Processing {imgfullpath}")

    img = cv2.imread(imgfullpath, cv2.IMREAD_GRAYSCALE)
    imgc = cv2.imread(imgfullpath, cv2.IMREAD_COLOR)
    #edges = cv2.Canny(img.copy(), 130, 250)
    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

    ret, thresh = cv2.threshold(img.copy(), 130, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f"{imgfullpath}-1-threshhold.png", thresh)

    # find contours in the thresholded image, then initialize the digit contours lists
    input_image_array, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #print("Number of Contours is: " + str(len(cnts)))

    digitCnts = []
    digitBdBx = []
    for i, c in enumerate(cnts):
        # compute the bounding box of the contour
        area = cv2.contourArea(c)
        br = cv2.boundingRect(c)
        x, y, w, h = br
        if (w >= width_min and w <= width_max) and (h >= height_min and h <= height_max):
        #if (w >= 25 and w <= 60)  and (h >= 60 and h <= 110):
            #print(w, h)
            digitCnts.append(c)
            digitBdBx.append(br)

    # Draw small black contour lines around all images in the original image to be used as a mask later
    img_contours = cv2.drawContours(imgc.copy(), digitCnts, -1, (0, 0, 0), 4)
    
    # sort the contours from left-to-right
    # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
    (digitCnts, digitBdBx) = zip(*sorted(zip(digitCnts, digitBdBx), key=lambda b:b[1][0], reverse=False))

    img_all_boxes = imgc.copy()
    digits = []
    digits_contour = []
    for i, (c, b) in enumerate(zip(digitCnts, digitBdBx)):
        x, y, w, h = b
        #print(f"Size: {x}, {y}, {w}, {h}")
        #print("---------------------------------")

        # Cut out bounding boxes from original image and create a single image from each digit
        digit = imgc[y:y + h, x:x + w]
        #digits.append(digit)
        #cv2.imwrite(f"{imgfullpath}-6-SingleDigit-{i}.png", digit)

        # Create a single contour image for each digit
        #mask = np.ones(img_contours.shape[:2], dtype="uint8") * 255
        #cv2.drawContours(mask, c, -1, (0, 0, 0), 3)
        #cv2.bitwise_and(img_contours, img_contours, mask=mask)
        #digit_cont = mask[y:y + h, x:x + w]
        #cv2.imwrite(f"{imgfullpath}-5-SingleDigitContour-{i}.png", digit_cont)
        #digits_contour.append(digit_cont)
        
        # Create a single threshhold image for each digit
        digittresh = thresh[y:y + h, x:x + w]
        cv2.imwrite(f"{feature_path}/SingleDigitThreshhold-{i}.png", digittresh)
        # Write single bounding box to original image
        #imgcbb = cv2.rectangle(imgc.copy(),(x,y),(x+w,y+h),(0,255,0),3)
        #cv2.imwrite(f"{imgfullpath}-4-BoundingBox-{i}.png", imgcbb)
        
        # Draw all bonding boxes into the original image
        #cv2.rectangle(img_all_boxes,(x,y),(x+w,y+h),(0,255,0),3)

    #cv2.imwrite(f"{imgfullpath}-3-BoundingBox.png", img_all_boxes)

    # Draw all contour for all digits into the original image
    #img_contours = cv2.drawContours(imgc.copy(), digitCnts, -1, (0, 255, 0), 3)
    #cv2.imwrite(f"{imgfullpath}-2-contours.png", img_contours)
    
class HDF5Writer(object):
    def __init__(self, dims, output_path, data_key="images", buf_size=1000):
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        self.bufSize = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()


# Pseudo code:
# ---------------------------------------------------------------
#data_dim = (num_of_images, x_dim, y_dim, num_of_colours)
#writer = HDF5Writer(data_dim, output_path)

#for image_path in paths:
#    image = read_image(image_path)
#    image = preprocess(image) 
#    writer.add([image], [label])

#writer.close()
# ---------------------------------------------------------------

#cams = [cv2.VideoCapture(0), cv2.VideoCapture(1)]
#cams = [0, 2]
#led = [LED(17), LED(27)]

cams = [2]
led = [LED(27)]

img_counter = 0

#try:
while True:
    timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    day = datetime.now().strftime("%Y.%m.%d")

    datadir = f"{img_dir}/{day}"
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        
    for i, cam in enumerate(cams):
        led[i].on()
        img_name = f'image_{timestamp}_cam{cam}.jpg'
        featuredir_name = f'{img_name}_features'
        #print(f'Saving image file {datadir}/{img_name}')
        uvcd_dev = f"--device=/dev/video{cam}"
        uvcd_cmd = [["--set=Focus, Auto", "0"],
                    ["--set=Exposure, Auto", "1"]]
                    
        fswc_cmd = ["--device", f'v4l2:/dev/video{cam}', 
                    "--input", "0",
                    "--resolution","2592x1944",
#                    "--delay", "1",
#                    "--skip", "10",
                    "--frames", "20",
                    "--jpeg", "95",
                    "--no-banner"]
        #fswc_cmd.extend(["--set", "Focus, Auto=False",
        #                 "--set", "Exposure, Auto=Manual Mode"])

        if cam == 0:
            # [--]x[|],[X]x[Y]
            img_path = f"{datadir}/cam{cam}"
            feature_path = f"{img_path}/{featuredir_name}"
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            imgfullpath = f"{img_path}/{img_name}"
            uvcd_cmd.extend([["--set=Focus (absolute)", "5"]])
            uvcd_cmd.extend([["--set=Exposure (Absolute)", "554"]])
            #fswc_cmd.extend(["--set", "Exposure (Absolute)=554"])
            #fswc_cmd.extend(["--set", "Focus (absolute)=5"])
            fswc_cmd.extend(["--crop", "749x197,703x1587"])
            #fswc_cmd.extend(["--rotate", "270"])
            #fswc_cmd.extend(["--greyscale"])
            fswc_cmd.extend(["--save", imgfullpath])

        if cam == 2:
            # [--]x[|],[X]x[Y]
            img_path = f"{datadir}/cam{cam}"
            feature_path = f"{img_path}/{featuredir_name}"
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            imgfullpath = f"{img_path}/{img_name}"
            #uvcd_cmd.extend([["--set=Focus (absolute)", "6"]])
            #uvcd_cmd.extend([["--set=Exposure (Absolute)", "554"]])
            #fswc_cmd.extend(["--set", "Exposure (Absolute)=554"])
            #fswc_cmd.extend(["--set", "Focus (absolute)=6"])
            fswc_cmd.extend(["--crop", "656x163,1122x1508"])
            #fswc_cmd.extend(["--rotate", "270"])
            #fswc_cmd.extend(["--greyscale"])
            fswc_cmd.extend(["--save", imgfullpath])
            
        #print(" ".join(uvcd_cmd))
        #print(" ".join(fswc_cmd))
        
        #uvcdynctrl = ['uvcdynctrl', uvcd_dev]
        #for item in uvcd_cmd:
        #    cmd = uvcdynctrl + item
        #    print(" ".join(cmd))
        #    setup_cam = subprocess.Popen(cmd,
        #                                stdout=subprocess.PIPE,
        #                                stderr=subprocess.PIPE,
        #                                text=True)
        #    setup_cam.wait()
        #    setup_cam_status = setup_cam.poll()
        #    setup_cam_stdout, setup_cam_stderr = setup_cam.communicate()
        #    print(f"---------------Setting {item}----------------")
        #    print(setup_cam_status)
        #    print(setup_cam_stdout)

        #check_items = [["--get=Focus, Auto"], ["--get=Focus (absolute)"], ["--get=Exposure, Auto"], ["--get=Exposure (Absolute)"]]
        #for item in check_items:
        #    print(uvcdynctrl)
        #    print(item)
        #    cmd = uvcdynctrl + item
        #    print(" ".join(cmd))
        #    setup_cam = subprocess.Popen(cmd,
        #                                stdout=subprocess.PIPE,
        #                                stderr=subprocess.PIPE,
        #                                text=True)
        #    setup_cam.wait()
        #    setup_cam_status = setup_cam.poll()
        #    setup_cam_stdout, setup_cam_stderr = setup_cam.communicate()
        #    print(f"---------------Checking {item}----------------")
        #    print(setup_cam_status)
        #    print(setup_cam_stdout)

        #time.sleep(10)

        fswebcam = ['fswebcam']
        get_img = subprocess.Popen(fswebcam + fswc_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
        get_img.wait()
        get_img_status = get_img.poll()
        get_img_stdout, get_img_stderr = get_img.communicate()
    
        led[i].off()
        st = time.time()
        detect_edge(img_name, imgfullpath, feature_path, width_min=25, width_max=60, height_min=60, height_max=110)
        et = time.time()
        #print(et - st)

    img_counter += 1

#except KeyboardInterrupt as e:
#    print("Terminating...")
#    print(e)
#    for l in led:
#        l.off()
#    sys.exit(0)
#finally:
#    for l in led:
#        l.off()
#    sys.exit(0)
