#!/bin/bash
#
#

for i in {0..6} ; do
	fswebcam --device v4l2:/dev/video0 --input 0 --set "Focus, Auto"=False --resolution 2592x1944 --delay 1 --skip 10 --frames 20 --jpeg 95 --no-banner --set "Focus (absolute)"=${i} --save /home/pi/wasser/images/focustest_cam0_${i}.jpg
	fswebcam --device v4l2:/dev/video2 --input 0 --set "Focus, Auto"=False --resolution 2592x1944 --delay 1 --skip 10 --frames 20 --jpeg 95 --no-banner --set "Focus (absolute)"=${i} --save /home/pi/wasser/images/focustest_cam1_${i}.jpg	
done
