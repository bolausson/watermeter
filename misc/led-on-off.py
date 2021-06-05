#!/usr/bin/env python3
#
#

import os
import sys
#import h5py
import time
import socket
import signal
import subprocess
from gpiozero import LED
#import RPi.GPIO as GPIO
from datetime import datetime

signal.signal(signal.SIGINT, signal.default_int_handler)

led = [LED(17), LED(27)]


#    time.sleep(3)
#    led[i].off()


try:
    # Use GPIO numbers not pin numbers
    #GPIO.setmode(GPIO.BCM)
    #GPIO.setwarnings(False)
    #GPIO.setup(17,GPIO.OUT)
    #GPIO.output(17,GPIO.HIGH)
    for i in [0, 1]:
        led[i].on()
    time.sleep(300000000)
    #GPIO.output(17,GPIO.LOW)
    for i in [0, 1]:
        led[i].off()

except (KeyboardInterrupt):
    print("Terminating...")
    #GPIO.output(17,GPIO.LOW)
    for i in [0, 1]:
        led[i].off()
    sys.exit(0)
finally:
    #GPIO.output(17,GPIO.LOW)
    for i in [0, 1]:
        led[i].off()
    sys.exit(0)
