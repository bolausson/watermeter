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
from termcolor import colored, cprint
from colorama import Fore, Back, Style
from itertools import cycle
from PIL import Image, ImageDraw, ImageFont, ImageColor

colornames = [c[0] for c in ImageColor.colormap.items()]
colorcycle = cycle(colornames)
textcycle = cycle(range(0,99,1))

def next_ct():
    nc = next(colorcycle)
    nt = next(textcycle)
    return nc, nt


files = glob.glob('./retrain_images/*/*/*_reshaped.png')
font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 30)

for f in files:
    nc, nt = next_ct()

    img = Image.new('RGB', (36,76), color = nc)
    d = ImageDraw.Draw(img)
    d.text((0,0), f"{nt}", font=font, fill='black')
 
    #f = f.replace(".png", "_test.png")
    img.save(f'{f}')
