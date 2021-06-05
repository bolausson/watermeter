#!/bin/bash
#
#

rsync -ah --remove-source-files --info=progress2,stats2 pi@meterpi:/media/pi/Bjoern/images/test_images /home/blub/wasser/training/
