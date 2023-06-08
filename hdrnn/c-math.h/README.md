# Handwritten Digit Recognizer using Neural Network in C

Note that the data set is not added to repo, find it online :)

Alternatively, run `get-mnist.sh`

## Compiling

Create a bin folder (`mkdir -p bin/`) in `hdrnn/c-math.h/`

Then use the ubiquitous autotools

`make`

The binary will be written to the `bin/` folder created above

## Generate Sample Images

Create an images folder (`mkdir -p images/`) in `hdrnn/c-math.h/`

Once the MNIST dataset is downloaded, run `make sample-image`

To generate a different image run,

```
./bin/genimage <INDEX_INTO_TEST_IMAGES> <OUTPUT_PGM_FILE>
```

Alternatively see this facility in `hdrnn/python-numpy`

## Network Description

See `hdrnn/README.md` for details about the network description file

## Usage Comments

Running the program without any arguments for a help message

### Inference

```
./bin/hdrnn infer <IMAGE> --net <NETWORK>
```

You can create images using the methods specified in this README.md

It is assumed that the image will be a kind of PGM (2) format

The paths to the weights and biases files must contain HDR-NN weights and biases created by the `python-numpy`, `c-math.h`, or `cpp-eigen` programs

### Training

```
./bin/hdrnn train --net <NETWORK>
```

Default value assumed for net as `c-math.nn`

#### Silent invocation

Use silent invocation using the `--quiet` or `-q` flag

When used with the `train` command, the final weights and biases files won't be generated

```
./bin/hdrnn train -q
```

### Parameters for the Learning Algorithm

#### Epochs

> `--epochs` `-e` N

The number of epochs to train the network

```
./bin/hdrnn train -e 21
```

#### Network Shape

> `--shape` `-s` N,+

The size and shape of the HDR-NN network as comma seperated numbers

```
./bin/hdrnn train --shape 16,16
```

#### Learning Rate

> `--learning_rate` `-lr` F

The learning rate for the Stochastic Gradient Descent as a floating point value

```
./bin/hdrnn train -lr 0.3
```

#### Batch Size

> `--batch_size` `-bs` N

The batch size while performing mSGD as an integer

```
./bin/hdrnn train -bs 64
```
