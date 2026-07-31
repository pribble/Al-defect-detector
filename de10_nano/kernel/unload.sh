#!/bin/sh
module="cmadrv"
rmmod $module.ko 2>/dev/null || true
rm -f /dev/${module}0
