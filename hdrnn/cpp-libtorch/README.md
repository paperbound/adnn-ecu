# mSGD ANN (MNIST) training benchmark using libtorch

Derived from example application provided by `torch`

Building this project requires the use of libtorch library and the MNIST dataset

See https://github.com/pytorch/pytorch#from-source for instructions on building PyTorch

## Compiling

Create a build directory (mkdir -p build), run `cmake` with path to libtorch

```
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```

```
cmake --build .
```

## Usage comments

Run the binary normally, `./mnist`

### Inference

```
./mnist infer <IMAGE> --net <NETWORK>
```

You can create images using the methods specified in this README.md

It is assumed that the image will be a kind of PGM (2) format

The paths to the weights and biases files must contain HDR-NN weights and biases created by the `python-numpy`, `c-math.h`, or `cpp-eigen` programs

### Training

```
./mnist train --net <NETWORK>
```

Default value assumed for net as `c-math.nn`

#### Silent invocation

Use silent invocation using the `--quiet` or `-q` flag

When used with the `train` command, the final weights and biases files won't be generated

```
./mnist train -q
```

### Parameters for the Learning Algorithm

#### Epochs

> `--epochs` `-e` N

The number of epochs to train the network

```
./mnist train -e 21
```

#### Network Shape

> `--shape` `-s` N,+

The size and shape of the HDR-NN network as comma seperated numbers

```
./mnist train --shape 16,16
```

#### Learning Rate

> `--learning_rate` `-lr` F

The learning rate for the Stochastic Gradient Descent as a floating point value

```
./mnist train -lr 0.3
```

#### Batch Size

> `--batch_size` `-bs` N

The batch size while performing mSGD as an integer

```
./mnist train -bs 64
```
