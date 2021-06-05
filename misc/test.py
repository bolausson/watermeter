#!/usr/bin/env python3
from __future__ import print_function
import sys
import cv2
import time

def main(argv, focussetting):
    #capture from camera at location 0
    cap = cv2.VideoCapture(0)
    # Change the camera setting using the set() function
    #cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 90)
    #cap.set(cv2.CAP_PROP_FOCUS, int(focussetting))
    #cap.set(cv2.CAP_PROP_EXPOSURE, 90)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    # cap.set(cv2.CAP_PROP_GAIN, 4.0)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 144.0)
    # cap.set(cv2.CAP_PROP_CONTRAST, 27.0)
    # cap.set(cv2.CAP_PROP_HUE, 13.0) # 13.0
    # cap.set(cv2.CAP_PROP_SATURATION, 28.0)
    # Read the current setting from the camera
    #test = cap.get(cv2.CAP_PROP_POS_MSEC)
    #ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
    #frame_rate = cap.get(cv2.CAP_PROP_FPS)
    #width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    #contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    #saturation = cap.get(cv2.CAP_PROP_SATURATION)
    #hue = cap.get(cv2.CAP_PROP_HUE)
    #gain = cap.get(cv2.CAP_PROP_GAIN)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    exposuretype = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    focustype = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    focus = cap.get(cv2.CAP_PROP_FOCUS)
    #print("Test: ", test)
    #print("Ratio: ", ratio)
    #print("Frame Rate: ", frame_rate)
    #print("Height: ", height)
    #print("Width: ", width)
    #print("Brightness: ", brightness)
    #print("Contrast: ", contrast)
    #print("Saturation: ", saturation)
    #print("Hue: ", hue)
    #print("Gain: ", gain)
    print("Exposure type: ", exposuretype)
    print("Exposure: ", exposure)
    print("Focus type: ", focustype)
    print(f"Focus: ({focussetting})", focus)

    for i in range(1,60,1):
        temp = cap.read()
    ret, img = cap.read()
    cv2.imwrite(f"/home/pi/wasser/error_images/test_focus-{focussetting:03}.jpg", img)
    #cv2.imshow("input", img)
    #cv2.waitKey(3000)

#    while True:
#        ret, img = cap.read()
#        cv2.imwrite("/home/pi/wasser/error_images/test_focus-{focussetting}.jpg", img)
        #cv2.imshow("input", img)

        #key = cv2.waitKey(10)
        #if key == 27:
        #    break

    #cv2.destroyAllWindows()
    cap.release()
    #cv2.VideoCapture(2).release()


#   0  CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
#   1  CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
#   2  CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
#   3  CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
#   4  CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
#   5  CV_CAP_PROP_FPS Frame rate.
#   6  CV_CAP_PROP_FOURCC 4-character code of codec.
#   7  CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
#   8  CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
#   9 CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
#   10 CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
#   11 CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
#   12 CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
#   13 CV_CAP_PROP_HUE Hue of the image (only for cameras).
#   14 CV_CAP_PROP_GAIN Gain of the image (only for cameras).
#   15 CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
#   16 CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
#   17 CV_CAP_PROP_WHITE_BALANCE Currently unsupported
#   18 CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

if __name__ == '__main__':
    for i in range(5, 100, 5):
        main(sys.argv, i)
        time.sleep(1)
