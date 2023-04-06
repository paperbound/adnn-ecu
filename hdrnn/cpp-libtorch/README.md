# mSGD ANN (MNIST) training benchmark using libtorch

Derived from example application provided by `torch`

## Compiling

Create a build directory, run cmake with path to libtorch

```
$ cd mnist
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

## Usage comments

Run the binary normally, `./mnist`
