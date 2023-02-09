/*
 * HDRNN
 * Handwritten Digit Recognition Neural Network
 *
 * @author Prasanth Thomas Shaji
 *
 */

#ifndef HDRNN_HDRNN_H
#define HDRNN_HDRNN_H

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer>
#include <vector>

#include "Eigen/Core"

namespace fs = std::filesystem;

using Eigen::MatrixXf;
using Eigen::RowVectorXf;

extern "C" {
	#include <mnist.h>
}

/* MNIST Dataset */
extern float train_images[NUM_TRAIN][SIZE];
extern float test_images[NUM_TEST][SIZE];
extern int train_labels[NUM_TRAIN];
extern int test_labels[NUM_TEST];

extern int info_image[LEN_INFO_IMAGE];
extern int info_label[LEN_INFO_LABEL];

/* Current Image */
extern float image[SIZE]; // input layer / loaded image

namespace hdrnn {

// TODO: Take this out to a configuration file
// CONFIG start
/* Hyperparameters */
int epochs = 30;
int batch_size = 10;
float eta = 3;

/* HDRNN shape */
#define INPUT_LAYER_SIZE 784
#define OUTPUT_LAYER_SIZE 10
// CONFIG end

/* Activation Functions */
float sigmoid(float)
{
	return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

/* HDR Neural Network (hdrnn) */
class hdrnn
{
public:	/* HDRNN API functions */

	/* HDRNN constructor */
	hdrnn (std::initializer_list<int>);

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
	void infer_image_from_path(fs::path) const;

private:

	/* Layer struct for each Network layer
	 *
	 * Neurons are represented by the structure
	 * of the contained Eigen objects
	 * weights - Matrix of Weights for each Neuron in the layer
	 * bias    - Vector of biases for each Neuron in the layer
	 */
	struct layer {
		MatrixXf *weights;
		RowVectorXf *bias;

	public:

		layer(dim)
		{
			weights = new MatrixXf(dim, dim);
			bias = new RowVectorXf(dim);
		}

		~layer()
		{
			delete weights;
			delete bias;
		}
	};

	/* Infer current image of a handwritten digit using the network */
	int infer_image()
	{
		feed_forward(image);
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

	void feed_forward(RowVectorXf &image)
	{
	}

	void back_propogate()
	{
	}

	unsigned int epochs = epochs;
	unsigned int batch_size = batch_size;
	float eta = eta;

	RowVectorXf image;
	vector<layer> network;
}

/* HDRNN constructor
 *
 * shape - specifies the number and dimension of the hidden layers to initialize
 */
hdrnn::hdrnn(std::initializer_list<int> shape)
{
	// iterate over the dimensions required in the hidden layer
	for (auto dim : shape)
	{
		network.push_back(layer(dim));
	}
	// finally, add the last layer
	network.push_back(layer(10));
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
		evaluate_network();
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
	ifstream w_stream, b_stream;

	// Open the weight file
	if (!w_stream.open(w_file))
	{
		std::cerr << "Could not open weights file" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// Open the bais file
	if (!b_stream.open(b_file))
	{
		std::cerr << "Could not open bias file" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	unsigned int w_dim1, w_dim2, b_dim;
	float x;

	// Read the file contents
	for (auto layer: network)
	{
		w_stream >> w_dim1 >> w_dim2;
		b_stream >> b_dim;

		// TODO: Add asserts here for w_dim1 == b_dim
		for (auto i: layer.bias->size())
		{
			
		}
	}
	
}

/* Infer the digit in an image from path using the network
 *
 * Note : Changes the current image
 *        Outputs the digit that the HDRNN thinks the image represents
 * i_file - 
 */
void hdrnn::infer_image_from_path(fs::path i_file)
{
	// File input stream for image file
	ifstream i_stream;

	// Open the weight file
	if (!w_stream.open(w_file))
	{
		std::cerr << "Could not open weights file" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	float 

}

} // namespace hdrnn

#endif // HDRNN_HDRNN_H
