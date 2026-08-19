#!/bin/sh
insmod cmadrv.ko
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH+
./ssd_detection config.txt data
