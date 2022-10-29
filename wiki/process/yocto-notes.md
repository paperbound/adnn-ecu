# Yocto Notes

Process notes on the Yocto workflow used in this project

## Summary

Applications housed within this repostory, such as those in the `hdrnn` directory, will have a direct image recipe in the `meta-ann` layer. Sections in this document will serve as guidelines to how these recipes should be written

## Background

### Setting up the Environment

The workflow for this project will start out using the poky reference distribution. As such the development environment will have directories such as poky, meta-ti, and other layers providing support. These will be accessed via git submodules as described in `version-control.md`.

To setup the environment run the following script found in the `scripts` directory

```bash
./scripts/run-project-setup.sh
```

Take a look at this file in the script directory to learn about how the layers are setup in this workflow

### Yocto Terminology

The Yocto project has what's called a "Layer Model". Layers are repositories that contain related sets of instructions that tell the OpenEmbedded Build System what to do. You can see some of the layers used in this repository as the submodules `poky`, `meta-ti`, and `meta-arm`. The repository also houses a layer called `meta-adnn`

## Creating an Image recipe

An image in Yocto are top level recipes that defines how the root file system is build and what packages it will contain. It has a description, a license, and inherits the core-image-class

An example of an important image recipe used in this project can be found in `meta-ann/recipes-core/images/ann-image-minimal.bb`

```bitbake
SUMMARY = "A small image with the <application>"

IMAGE_INSTALL = "packagegroup-core-boot <application>"

LICENSE = "MIT"

inherit core-image
```

To create an image with this recipe, run:

```bash
bitbake ann-image-minimal
```

Note that the current LICENSE values through out `meta-adnn` are placeholders

## Developing Applications

Yocto provides recipes for setting up a development environment for the target device. This is activated using the `populate_sdk` task that comes with the `ann-image-minimal` image recipe used in this workflow

```bash
bitbake -c populate_sdk ann-image-minimal
```

This task will ask where to place the SDK which for this repository we will assume to be in `sdk` directory which is not checked in

### Creating an application recipe

The details of how to compile and install the application will be described in an application recipes in the `meta-adnn/recipes-adnn` directory

An example application recipe can be found in `meta-adnn/recipes-adnn/hdrnn-cmath/hdrnn-cmath_0.1.bb`

### Running the application on the device

Get the rootfs to the machine. This can be done through multiple means depending on the hardware.

Current setup for development using the beagle bone involves using the kernel boot arguments send by uboot setting up an NFS folder as rootfs. The rootfs image is hence just extracted to this nfs location on the development device while the beaglebone is connected to it.

### Creating a Machine Configuration

_This part will be added in later while developing a machine configuration for the C300_
