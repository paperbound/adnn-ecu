#!/usr/bin/env sh

#
# Script to setup the project submodules
# @author Prasanth Shaji
#
# Submodules that are setup in this script are,
# 1. poky from https://git.yoctoproject.org/poky/
# 2. meta-arm from https://git.yoctoproject.org/meta-arm/
# 3. meta-ti from https://git.yoctoproject.org/meta-ti/
#
# For all modules kirkstone branch will be chosen
# specifically, kirkstone 4.0.2
#
# NOTE: Additional modules to follow
#

# Setup the submodules
cd `git rev-parse --show-toplevel`
git checkout main
git submodule update --init

# Setup poky

cd poky
git checkout -b kirkstone-4.0.2 kirkstone-4.0.2

# Setup meta-arm

cd `git rev-parse --show-toplevel`
cd meta-arm
git checkout -b yocto-4.0 yocto-4.0

# Setup meta-ti

cd `git rev-parse --show-toplevel`
cd meta-ti
git checkout -b kirkstone kirkstone

# Setup bitbake
cd `git rev-parse --show-toplevel`
source poky/oe-init-build-env

# Setup layers
bitbake-layers add-layer meta-ti
bitbake-layers add-layer meta-arm
bitbake-layers add-layer meta-adnn
