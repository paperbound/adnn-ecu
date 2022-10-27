SUMMARY = "A small bootable image with a basic nn application"

IMAGE_INSTALL = "packagegroup-core-boot ${CORE_IMAGE_EXTRA_INSTALL} hdrnn-cmath"

IMAGE_LINGUAS = ""

LICENSE = "MIT"

inherit core-image
