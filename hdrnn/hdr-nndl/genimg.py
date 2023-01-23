# Generate PGM images from the MNIST test data set
# @author Prasanth Shaji
#
# Please see README.md before running

# Load the data set
import numpy as np

import mnist_loader
_, _, test_data = mnist_loader.load_data_wrapper()
test_data = list(test_data)

# Write an image
import sys


test_index = sys.argv[1]

try:
	assert test_index.isnumeric()
	np.set_printoptions(formatter={'float': '{: 2.2f}'.format})
	with open(test_index + ".pgm", mode='w') as file:
		file.write("P2\n")
		file.write("28 28\n")
		file.write("255\n")
		count = 0
		for x in test_data[int(test_index)][0]:
			file.write(str(float(x))[:5] + ', ')
			count += 1
			if (count % 28 == 0):
				file.write('\n')

except OSError as err:
	print("OS error:", err)
except AssertionError as err:
	print("Please enter an integer index")
