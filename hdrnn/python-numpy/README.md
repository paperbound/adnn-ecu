# Neural Networks and Deep Learning

The implementation of a fully connected neural network as described in the [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) book

## Usage

Activate an anaconda base environment by running `conda activate base`

### Basic Usage

1. Train the HDRNN using the command `python3 train.py`.

2. Generate an image file for a test image from MNIST dataset using `python3 genimg.py TEST_IMAGE_INDEX`. e.g : `python3 genimg.py 4` will create an image file `4.pgm`

3. Infer an MNIST image created above using `python3 infer.py -i IMAGE`

### Training

By default the training outputs the weights into file `numpy.nn` controlled by `--net` or `-n` argument

The parameters for the learning algorithm are:

#### Epochs

> `--epochs` `-e` N

The number of epochs to train the network

```
python3 train.py --epochs 10
```

#### Network Shape

> `--shape` `-s` N,+

The size and shape of the HDR-NN network as comma seperated numbers

```
python3 train.py --shape 16,16
```

#### Learning Rate

> `--learning_rate` `-lr` F

The learning rate for the Stochastic Gradient Descent as a floating point value

```
python3 train.py --learning_rate 3.1
```

#### Batch Size

> `--batch_size` `-bs` N

The batch size while performing mSGD as an integer

```
python3 train.py -bs 64
```
