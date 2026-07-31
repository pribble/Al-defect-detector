# Kernel Module Makefile — cmadrv (CMA + DMA driver for Cyclone V)
#
# Cross-compile from x86_64 host for arm target.
# Usage: make all          # build cmadrv.ko
#        make clean        # clean build artifacts

# Kernel source tree (path to the built kernel on host)
# KSRC_DIR must point to a cross-compiled kernel tree.
KSRC_DIR ?= /opt/software/linux-4.9.78

# Architecture and cross compiler prefix
ARCH := arm
CROSS_COMPILE := arm-linux-gnueabihf-

# Called from kernel build system — the actual build happens here
ifneq ($(KERNELRELEASE),)
  obj-m := cmadrv.o

# Called from command line — enter kernel build system
else
  PWD := $(shell pwd)

all:
	@$(MAKE) -C $(KSRC_DIR) M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) modules

clean:
	@rm -rf *.ko *.o *~ .*.cmd *.mod.c .tmp_versions Module.symvers modules.order

.PHONY : all clean
endif
