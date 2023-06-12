# Infer an image using HDR-NN
# @author Prasanth Shaji
#
# Please see README.md before running

# Command line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', dest='ifile',
                    action='store', default='1.pgm')
parser.add_argument('-n', '--net', dest='nfile',
                    action='store', default='numpy.nn')
args = parser.parse_args()

ifile = args.ifile
nfile = args.nfile

# Load the image
import numpy as np

a = []

with open(ifile, 'r') as f:
    f.readline() # P2
    f.readline() # 28 28
    f.readline() # 255
    for i in range(784):
        a.append(int(f.readline()) / 255.0)

a = np.reshape(a, (784, 1))

# Load the data set
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# Configure the network
import network
import struct
import sys

shape = [784] # input dimensions
biases = []
weights = []

bo = sys.byteorder

with open(nfile, 'rb') as f:
    f.read1(1)
    sizes = int.from_bytes(f.read1(1), bo)
    for s in range(sizes):
        shape.append(int.from_bytes(f.read1(1), bo))
    shape.append(10) # output layer
    for x in shape[1:]:
        lbiases = []
        for y in range(x):
            lbiases.append(struct.unpack('f', f.read1(4)))
        biases.append(np.reshape(lbiases, (x, 1)))
    for x, y in zip(shape[:-1], shape[1:]):
        lweights = []
        for i in range(x*y):
            lweights.append(struct.unpack('f', f.read1(4)))
        weights.append(np.reshape(lweights, (y, x)))

net = network.Network(shape)

if net.load_weights(biases, weights):
    print("Network gets : {1} / {2} correct".format(
                        self.evaluate(test_data), n_test))
    print("Network things image is an : ",
          np.argmax(net.feedforward(a)))
else:
    raise("Could not load the weights")
