# benchmark hdrnn

SIZES=("2" "4" "8" "32" "16,16" "128" "32,16")
COMMAND="train"
FLAGS="--epochs 1 --quiet"

## c-math.h

/usr/bin/time -v -- ./hdrnn/c-math.h/bin/hdrnn ${COMMAND} ${FLAGS}

## cpp-libtorch

/usr/bin/time -v -- ./hdrnn/cpp-libtorch/build/mnist ${COMMAND} ${FLAGS}

## cpp-eigen

/usr/bin/time -v -- ./hdrnn/cpp-eigen/bin/hdr ${COMMAND} ${FLAGS}

## python-numpy

/usr/bin/time -v -- python3 ./hdrnn/python-numpy/train.py ${FLAGS}
