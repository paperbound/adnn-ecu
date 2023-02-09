# Neural Networks and Deep Learning

The implementation of a fully connected neural network as described in the [Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning) book

## Usage

Activate an anaconda base environment by running `conda activate base`

1. Train the HDRNN using the command `python3 run.py`. This will generate a file `bias` and a file `weight` with the biases and weights of the final trained network

2. Generate an image file for a test image from MNIST dataset using `python3 genimg.py TEST_IMAGE_INDEX`. e.g : `python3 genimg.py 4` will create an image file `4.pgm`
