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

### Creating an Image recipe

`meta-ann/recipes-core/images/ann-image-minimal.bb`

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

### Developing Applications

Yocto provides recipes for setting up a development environment for the target device. This is activated using the `populate_sdk` task that comes with the `ann-image-minimal` image recipe that we use in this workflow

```bash
bitbake -c populate_sdk ann-image-minimal
```

### Running the application on the device

Get the rootfs to the machine. This can be done through multiple means depending on the hardware.

Current setup for development using the beagle bone involves using the kernel boot arguments send by uboot setting up an NFS folder as rootfs. The rootfs image is hence just extracted to this nfs location on the development device while the beaglebone is connected to it.

### Creating a Machine Configuration

_This part will be added in later while developing a machine configuration for the C300_
