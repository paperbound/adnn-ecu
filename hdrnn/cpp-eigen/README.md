# Handwritten Digit Recognition using Neural Network in CPP + Eigen

Note that the dataset is not added to the repo, find it online

Alternatively run the `get-mnist.sh` script in `hdrnn/c-math.h`

If the folder is already setup somewhere, the path to the folder maybe
specified while training the model after the `-train` flag. See more
in Usage

## Compiling

Please note that you need to create a bin folder (`mkdir -p bin`)
inside `hdrnn/cpp-eigen/` much like in `hdrnn/c-math.h/`

This project uses cmake

Firstly, create a build directory. E.g. `mkdir build`. Further commands
to run while building the project should be run in this build folder

Specify the source directory to cmake: `cmake <path_to_source>`, where
path_to_source is the `hdrnn/cpp-eigen/` directory containing the
CMakeLists.txt file

Now you can use autotools (cmake default) to build the project:
`cmake --build .`

The binary will be written to the `bin/` folder created in the project
source directory

## Usage Comments

For some helpful usage comments, you may run the program with a `-h` flag

The usage is similar to that described in `hdrnn/c-math.h/README.md`

### Inference

```
../bin/hdr infer <IMAGE> --net <NETWORK>
```

You can create images using the methods specified in this README.md

It is assumed that the image will be a kind of PGM (2) format

The paths to the weights and biases files must contain HDR-NN weights and biases created by the `python-numpy`, `c-math.h`, or `cpp-eigen` programs

### Training

```
../bin/hdr train --net <NETWORK>
```

Default value assumed for net as `c-math.nn`

#### Silent invocation

Use silent invocation using the `--quiet` flag

When used with the `train` command, the final weights and biases files won't be generated

```
../bin/hdr train --quiet
```

### Parameters for the Learning Algorithm

#### Epochs

> `--epochs` N

The number of epochs to train the network

```
../bin/hdr train --epochs 21
```

#### Network Shape

> `--shape` N,+

The size and shape of the HDR-NN network as comma seperated numbers

```
../bin/hdr train --shape 16,16
```

#### Learning Rate

> `--learning_rate` F

The learning rate for the Stochastic Gradient Descent as a floating point value

```
../bin/hdr train --learning_rate 0.3
```

#### Batch Size

> `--batch_size` N

The batch size while performing mSGD as an integer

```
../bin/hdr train --batch_size 64
```
