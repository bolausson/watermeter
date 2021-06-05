#!/bin/bash
#
#

rsync -ah --remove-source-files --info=progress2,stats2 pi@openhab:/home/pi/watermeter/images/retrain_images .
