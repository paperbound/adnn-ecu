# HDRNN as described in http://neuralnetworksanddeeplearning.com/chap1.html
# @author Michael Nielsen (primary)
# @author Prasanth Shaji
#
# Please see README.md before running

# Command line arguments
import argparse

parser = argparse.ArgumentParser(description="HDRNN from neuralnetworksanddeeplearning.com")
parser.add_argument('-e', '--epochs', dest='epochs',
                    action='store', default=30, type=int)
parser.add_argument('-s', '--shape', dest='shape',
                    action='store', default='30')
parser.add_argument('-lr', '--learning_rate', dest='lrate',
                    action='store', default=3.0, type=float)
parser.add_argument('-bs', '--batch_size', dest='bs',
                    action='store', default=10, type=int)
parser.add_argument('-n', '--net', dest='nfile',
                    action='store', default='numpy.nn')
parser.add_argument('-q', '--quiet', dest='quiet',
                    action='store_true', default=False)
a = parser.parse_args()

epochs = a.epochs
shape  = a.shape
lrate  = a.lrate
bs     = a.bs
nfile  = a.nfile
quiet  = a.quiet

# Load the data set
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# Configure and train the network
import network

# get the network shape
sizes = [784] # input dimension
try:
    sizes.extend([ int(n) for n in shape.split(',') ])
except:
    raise("Network shape specified incorrectly")
sizes.append(10) # output dimension

net = network.Network(sizes)

net.SGD(training_data, epochs, bs, lrate, test_data=test_data, quiet=quiet)

# Dump weights
if (not quiet):
    net.dump_weights(nfile)
