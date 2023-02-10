/*
 * HDRNN
 * Handwritten Digit Recognition Neural Network
 *
 * @author Prasanth Thomas Shaji
 *
 */

#ifndef HDRNN_HDRNN_H
#define HDRNN_HDRNN_H

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <vector>

#include "Eigen/Core"

namespace fs = std::__fs::filesystem;

using Eigen::Matrix;
using Eigen::MatrixXf;
using Eigen::VectorXf;

// TODO: Take this out to a configuration file
// CONFIG start
/* Hyperparameters */
const int EPOCHS = 30;
const int BATCH_SIZE = 10;
const float ETA = 3;

/* HDRNN shape */
#define INPUT_LAYER_SIZE 784
#define OUTPUT_LAYER_SIZE 10
// CONFIG end

/* Activation Functions */
float sigmoid(float x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

float sigmoid_prime(float x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

void display_vector(VectorXf &v)
{
	for (float a : v)
		std::cout << a << " ";
	std::cout << std::endl;
}

/* HDR Neural Network (hdrnn) */
class hdrnn
{
public:	/* HDRNN API functions */

	/* HDRNN constructor */
	hdrnn (std::initializer_list<unsigned int>);

	/* HDRNN destructor */
	~hdrnn ();

	/* Train the HDR neural network */
	void train_hdrnn();

	/* Evaluate the accuracy of the hdrnn */
	void evaluate_hdrnn();

	/* Dump csv file of weights and biases */
	void dump_hdrnn(fs::path, fs::path) const; // takes 2 path names

	/* Loads a csv file of weights and biases into the Network */
	void load_hdrnn(fs::path, fs::path);

	/* Infer the image at path of a handwritten digit using the network */
	void infer_pgm_image_from_path(fs::path);


private:

	/* Layer struct for each Network layer
	 *
	 * Neurons are represented by the structure
	 * of the contained Eigen objects
	 * weights - Matrix of Weights for each Neuron in the layer
	 * bias    - Vector of biases for each Neuron in the layer
	 */
	struct layer {
		MatrixXf weights;
		VectorXf bias;

		layer(unsigned int c_dim, unsigned int p_dim) {
			weights = MatrixXf(c_dim, p_dim);
			bias = VectorXf(c_dim);
		}
	};

	/* Infer current image of a handwritten digit using the network */
	int infer_image()
	{
		VectorXf a = feed_forward();
		float best = -1;
		int id = -1;
		for (std::size_t k = 0; k < a.size(); k++)
		{
			if (best < a(k))
			{
				best = a(k);
				id = k;
			}
		}
		display_vector(a);
		return id;
	}

	/* Initialize random weights and biases in the network */
	void generate_random_weights()
	{
	}

	/* Mini Batched Stochastic Gradient Descend Algorithm
	 *
	 * Reference implementation from http://neuralnetworksanddeeplearning.com
	 */
	void mini_batch_sgd()
	{
	}

	VectorXf feed_forward()
	{
		VectorXf activations = image;
		for (std::size_t k = 0; k < network.size(); k++)
			activations = ((network[k].weights * activations)
				       + network[k].bias)
				.unaryExpr(&sigmoid);
		return activations;
	}

	void back_propogate()
	{
	}

	unsigned int epochs = EPOCHS;
	unsigned int batch_size = BATCH_SIZE;
	float eta = ETA;

	Matrix<float, 1, INPUT_LAYER_SIZE> image;
	std::vector<layer> network;
};

/* HDRNN constructor
 *
 * shape - specifies the number and dimension of the hidden layers to initialize
 */
hdrnn::hdrnn(std::initializer_list<unsigned int> shape)
{
	// size of image in row vector form
	unsigned int previous_dim = INPUT_LAYER_SIZE;

	// iterate over the dimensions required in the hidden layer
	for (auto dim : shape)
	{
		network.push_back(layer(dim, previous_dim));
		previous_dim = dim;
	}
	// finally, add the last layer
	network.push_back(layer(OUTPUT_LAYER_SIZE, previous_dim));
}

/* HDRNN destructor */
hdrnn::~hdrnn()
{
	// TODO : figure out if the layer destructors gets called at this point
	network.clear();
}

/* Performs Mini Batched - Stochastic Gradient Descent to train the HDRNN
 *
 * Notes: Replaces the weights in the current network
 *        Displays the accuracy of the network after each epoch
 */
void hdrnn::train_hdrnn()
{
	for (int i = 0; i < epochs; i++)
	{
		mini_batch_sgd();
		evaluate_hdrnn();
	}
}

/* Evaluate the HDRNN by using the MNIST test dataset
 *
 * Note : Prints out the current accuracy of the network
 */
void hdrnn::evaluate_hdrnn()
{
}

/* Loads files with weights and biases to update the same on the network
 *
 * Note : Creates files with the weights and biases of the network
 */
void hdrnn::dump_hdrnn(fs::path w_file, fs::path b_file) const
{
}

// TODO: Create a better standard format for weights, biases files

/* Loads csv files with weights and biases to update the same on the network
 *
 * Note : Updates the weights and biases on the network
 */
void hdrnn::load_hdrnn(fs::path w_file, fs::path b_file)
{
	// File input streams for weights and biases file
	std::ifstream w_stream, b_stream;

	// Open the weight file
	w_stream.open(w_file);
	if (!w_stream)
	{
		std::cerr << "Could not open weights file" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Open the bias file
	b_stream.open(b_file);
	if (!b_stream)
	{
		std::cerr << "Could not open bias file" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	unsigned int w_dim1, w_dim2, b_dim;
	float x;

	// Read the file contents
	for (std::size_t k = 0; k < network.size(); k++)
	{
		w_stream >> w_dim1 >> w_dim2;
		b_stream >> b_dim;

		// TODO: Add asserts here for w_dim1 == b_dim
		for (unsigned int i = 0; i < w_dim1; i++)
		{
			b_stream >> x;
			network[k].bias(i) = x;
			for (unsigned int j = 0; j < w_dim2; j++)
			{
				w_stream >> x;
				network[k].weights(i, j) = x;
			}
		}
	}

	// close file streams
	w_stream.close();
	b_stream.close();
}

// TODO: Change to a more tradition PGM read approach
/* Infer the digit in an image from path using the network
 *
 * Note : Changes the current image
 *        Outputs the digit that the HDRNN thinks the image represents
v * i_file - filename of a PGM image (in a particular format)
 */
void hdrnn::infer_pgm_image_from_path(fs::path i_file)
{
	// File input stream for image file
	std::ifstream i_stream;

	// Open the weight file
	i_stream.open(i_file);
	if (!i_stream)
	{
		std::cerr << "Could not open image file" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	float x;

	// seek to position 3 to ignore PGM boilerplate
	i_stream.seekg(3);

	// read the image file into hdrnn image
	for (unsigned int i = 0; i < INPUT_LAYER_SIZE; i++)
	{
		i_stream >> x;
		image(i) = x / 255;
	}

	std::cout << infer_image() << std::endl;
}

#endif // HDRNN_HDRNN_H
