# General Overview

This project aims to run several neural networks applications on multiple embedded hardware. The Yocto project will be used to create images that house these neural network applications. To get a overview on how the Yocto project will be used, please refer to `yocto-notes.md`.

Work on this repository will follow the guidelines stated in `version-control.md`

## Current Hardware

List of hardware that the project currently targets are:

| Device           | Processor     |
| ---------------- | ------------- |
| Beaglebone black | ARM Cortex A8 |
| Raspberry Pi     | ARM Cortex A8 |

The Scania C300 will also be added later into this list

## Current Applications

The applications presently in this repository that run on at least one of the target devices mentioned above are:

### HDRNN

| Application | Location | Description |
| ----------- | -------- | ----------- |
| HDR cmath | `hdrnn/c-math.h`| FC, 1 HL, feedforward using clib math |

A fully connected, 1 hidden layer feedforward neural network application using the cmath library that performs only inference, along with associated weight files are present in the `hdrnn/c-math.h` directory
