/*
 * MNIST - HDRNN
 * @author Prasanth Thomas Shaji
 *
 * MNIST Data Loader
 * see _reference_ : http://yann.lecun.com/exdb/mnist/
 *
 */

#ifndef HDRNN_MNIST_H
#define HDRNN_MNIST_H

#include <iostream>
#include <fstream>

#include "Eigen/Core"

using Eigen::VectorXf;

/* Location for MNIST Data */
// TODO : maybe change out the #defines to something else
#define TRAIN_IMAGE "train-images-idx3-ubyte"
#define TRAIN_LABEL "train-labels-idx1-ubyte"
#define TEST_IMAGE "t10k-images-idx3-ubyte"
#define TEST_LABEL "t10k-labels-idx1-ubyte"

namespace mnist_loader {

	/* Image Parameters */
	const unsigned int IMAGE_SIZE = 784; // 28 x 28
	const unsigned int DIMENSION = 28;
	const unsigned int DIGITS = 10;

	const unsigned int TRAINING_SIZE = 50000;
	const unsigned int VALIDATE_SIZE = 10000;
	const unsigned int TESTING_SIZE  = 10000;

	const unsigned int IMAGE_INFO_LEN = 4;
	const unsigned int LABEL_INFO_LEN = 2;

	const unsigned int MAX_BRIGHTNESS = 255;

	// TODO: Consider storing as unsigned int as well
	struct image {
		VectorXf data;
		unsigned int label;
	};

	/* Training and Testing Data */
	std::vector<image> train;
	std::vector<image> validation;
	std::vector<image> test;

	/* Read MNIST Images from the dataset directory,
	 * Load images and labels for testing and training
	 *
	 * Note : assumes that the files will be present
	 * @param dd_path - the directory path containing the dataset files
	 * @return success - true or false value based on if the operation succeeded
	 */
	bool read_mnist(std::string dd_path)
	{
		// Input file stream for dataset files
		std::ifstream train_img_s, train_label_s, test_img_s, test_label_s;

		// Open the files
		train_img_s.open   (dd_path + TRAIN_IMAGE, std::ios::binary);
		train_label_s.open (dd_path + TRAIN_LABEL, std::ios::binary);
		test_img_s.open    (dd_path + TEST_IMAGE,  std::ios::binary);
		test_label_s.open  (dd_path + TEST_LABEL,  std::ios::binary);

		// Quit if any of the files are not available
		if (
		    !train_img_s
		    || !train_label_s
		    || !test_img_s
		    || !test_label_s
		   )
			return false;

		char x;

		// Ignore the info arrays
		train_img_s.seekg(IMAGE_INFO_LEN * sizeof(int));
		train_label_s.seekg(LABEL_INFO_LEN * sizeof(int));
		test_img_s.seekg(IMAGE_INFO_LEN * sizeof(int));
		test_label_s.seekg(LABEL_INFO_LEN * sizeof(int));

		// Read-in MNIST numbers (pixels|labels)
		// Read in training data
		for (std::size_t i = 0; i < TRAINING_SIZE; i++)
		{
			image img;
			img.data = VectorXf(IMAGE_SIZE);
			for (std::size_t j = 0; j < IMAGE_SIZE; j++)
			{
				train_img_s.read(&x, sizeof x);
				img.data[j] = static_cast<float>(x) / MAX_BRIGHTNESS;
			}
			train_label_s.read(&x, sizeof x);
			img.label = static_cast<unsigned int>(x);
			train.push_back(img);
		}
		// Read in test data
		for (std::size_t i = 0; i < TESTING_SIZE; i++)
		{
			image img;
			img.data = VectorXf(IMAGE_SIZE);
			for (std::size_t j = 0; j < IMAGE_SIZE; j++)
			{
				test_img_s.read(&x, sizeof x);
				img.data[j] = static_cast<float>(x) / MAX_BRIGHTNESS;
			}
			test_label_s.read(&x, sizeof x);
			img.label = static_cast<unsigned int>(x);
			test.push_back(img);
		}

		train_img_s.close();
		train_label_s.close();
		test_img_s.close();
		test_label_s.close();

		return true;
	}

} // mnist_loader

#endif
