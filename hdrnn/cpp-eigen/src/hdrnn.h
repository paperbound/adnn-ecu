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
#include <fstream>
#include <initializer_list>
#include <vector>

#include "Eigen/Core"

#include "mnist.h"

using Eigen::Index;
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

/* HDR Neural Network (hdrnn) */
class hdrnn
{
public:	/* HDRNN API functions */

	/* HDRNN constructor */
	hdrnn (std::initializer_list<unsigned int>);

	/* HDRNN destructor */
	~hdrnn ();

	/* Train the HDR neural network */
	void train_hdrnn(std::string);

	/* Evaluate the accuracy of the hdrnn */
	void evaluate_hdrnn(unsigned int);

	/* Dump csv file of weights and biases */
	void dump_hdrnn(std::string, std::string) const; // takes 2 path names

	/* Loads a csv file of weights and biases into the Network */
	void load_hdrnn(std::string, std::string);

	/* Infer the image at path of a handwritten digit using the network */
	void infer_pgm_image_from_path(std::string);

private:

	/* Nabla struct for SGD updates
	 *
	 * Nabla contains the ..........
	 * TODO: fill with an accurate description of the partial derivatvies
	 */
	struct nabla {
		VectorXf bias;
		MatrixXf weights;

		nabla(unsigned int c_dim, unsigned int p_dim)
		{
			bias = VectorXf::Zero(c_dim);
			weights = MatrixXf::Zero(c_dim, p_dim);
		}

		void zero_out()
		{
			bias = VectorXf::Zero(bias.rows());
			weights = MatrixXf::Zero(weights.rows(), weights.cols());
		}

		void accumulate(nabla& n)
		{
			// TODO : Add Asserts here
			bias += n.bias;
			weights += n.weights;
		}
	};

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

		layer(unsigned int c_dim, unsigned int p_dim)
		{
			weights = MatrixXf::Random(c_dim, p_dim);
			bias = VectorXf::Random(c_dim);
		}

		/* Update the weights and bias using the nabla
		 *
		 * @param n - Nablas with which to update
		 */
		void update(nabla& n)
		{
			float factor = ETA / BATCH_SIZE;
			weights = weights - (factor * n.weights);
			bias = bias - (factor * n.bias);
		}
	};

	/* Infer current image of a handwritten digit using the network
	 *
	 * @param image - Column vector of image values to infer
	 * @returns max - argmax of the output layer of neurons */
	Index predict(VectorXf& image)
	{
		Index max;
		VectorXf activation = feed_forward(image);
		activation.maxCoeff(&max);
		return max;
	}

	/* Initialize random weights and biases in the network */
	void generate_random_weights()
	{
		// TODO: Add a version that uses pcg_random
	}

	/* Mini Batched Stochastic Gradient Descend Algorithm
	 *
	 * Reference implementation from
	 * http://neuralnetworksanddeeplearning.com
	 */
	void mini_batch_sgd()
	{
		// Initialize Nabla matrixes
		std::vector<nabla> nablas;
		for (std::size_t i = 0; i < network.size(); i++)
			nablas.push_back(
					 nabla(network[i].weights.rows(),
					       network[i].weights.cols())
					 );

		// Go through the training data by batches
		for (std::size_t i = 0; i < mnist_loader::train.size()
			     ; i += BATCH_SIZE)
		{
			// Zero out the nabla matrixes
			for (std::size_t j = 0; j < nablas.size(); j++)
				nablas[j].zero_out();

			// Perform Backpropagation on the batch
			for (std::size_t j = 0; j < BATCH_SIZE; j++)
				back_propogate(nablas,
					       mnist_loader::train[i+j].data,
					       mnist_loader::train[i+j].label);

			// Update the weights and biases of the network
			for (std::size_t j = 0; j < network.size(); j++)
				network[j].update(nablas[j]);
		}
	}

	/* Feed Forward run
	 *
	 * @param image - Column vector of image values to be infered
	 * @returns activations - Output layer of HDRNN */
	VectorXf feed_forward(VectorXf &image)
	{
		VectorXf activations = image;
		for (std::size_t k = 0; k < network.size(); k++)
			activations = ((network[k].weights * activations)
				       + network[k].bias)
				.unaryExpr(&sigmoid);
		return activations;
	}

	/* Back Propagation run
	 *
	 * @param batach_nablas - Vector of nablas to update
	 * @param image  - Image to perform forward pass on
	 * @param label  - Label of passed image
	 * @param updates - Nabla values for bias and weight updates
	 */
	void back_propogate(
			    std::vector<nabla>& batch_nablas,
			    VectorXf &image, unsigned int label
			    )
	{
		// Initialize Nabla matrixes and activations
		std::vector<nabla> nablas;
		std::size_t num_layers = network.size() + 1;
		for (std::size_t i = 0; i < network.size(); i++)
			nablas.push_back(
					 nabla(network[i].weights.rows(),
					       network[i].weights.cols())
					 );

		std::vector<VectorXf> activations;
		VectorXf activation = image;
		activations.push_back(activation);
 
		// Feed forward
		std::vector<VectorXf> zs;
		VectorXf z;
		for (std::size_t i = 0; i < network.size(); i++)
		{
			z = (network[i].weights * activation)
				+ network[i].bias;
			zs.push_back(z);
			activation = z.unaryExpr(&sigmoid);
			activations.push_back(activation);
		}

		// Backward pass
		VectorXf y = VectorXf::Zero(OUTPUT_LAYER_SIZE);
		y[label] = 1;
		VectorXf delta = (activation - y)
			.cwiseProduct(z.unaryExpr(&sigmoid_prime));
		nablas[num_layers-2].bias += delta;
		nablas[num_layers-2].weights += (delta
					       * activations[num_layers-2]
					       .transpose());

		VectorXf sp;
		for (std::size_t i = num_layers-2; i > 0; i--)
		{
			z = zs[i-1];
			sp = z.unaryExpr(&sigmoid_prime);
			delta = (network[i].weights.transpose() * delta)
				.cwiseProduct(sp);
			nablas[i-1].bias += delta;
			// TODO : fix the i-1 in activations
			nablas[i-1].weights += (delta
						* activations[i-1]
						.transpose());
		}

		// Accumulate the new nablas
		for (std::size_t i = 0; i < batch_nablas.size(); i++)
			batch_nablas[i].accumulate(nablas[i]);
	}

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
void hdrnn::train_hdrnn(std::string dataset)
{
	mnist_loader::read_mnist(dataset);
	// TODO: Generate PCG-based weights and bias initializations
	for (unsigned int i = 0; i < EPOCHS; i++)
	{
		std::random_device r;
		std::default_random_engine e1(r());
		// std::shuffle(
		//      std::begin(mnist_loader::train),
		//      std::end(mnist_loader::train), e1
		//     );
		mini_batch_sgd();
		evaluate_hdrnn(i);
	}
}

/* Evaluate the HDRNN by using the MNIST test dataset
 *
 * Note : Prints out the current accuracy of the network
 */
void hdrnn::evaluate_hdrnn(unsigned int epoch)
{
	unsigned int count = 0;
	for (std::size_t i = 0; i < mnist_loader::test.size(); i++)
		if (
		    static_cast<unsigned int>(
					      predict(mnist_loader::test[i].data))
		    == mnist_loader::test[i].label
		    )
			count += 1;
	std::cout << "Epoch : " << epoch
		  << " Network has classified "
		  << count << "/"
		  << mnist_loader::test.size()
		  << " correctly" << std::endl;
}

/* Loads files with weights and biases to update the same on the network
 *
 * Note : Creates files with the weights and biases of the network
 */
void hdrnn::dump_hdrnn(std::string w_file, std::string b_file) const
{
	// File output streams for weights and biases file
	std::ofstream w_stream, b_stream;

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

	// Write the file contents
	for (std::size_t k = 0; k < network.size(); k++)
	{
		std::size_t w_dim1 = network[k].weights.rows();
		std::size_t w_dim2 = network[k].weights.cols();
		std::size_t b_dim  = network[k].bias.cols();

		w_stream << w_dim1 << std::endl
			 << w_dim2 << std::endl;
		b_stream << b_dim << std::endl;

		// TODO: Add asserts here for w_dim1 == b_dim
		for (unsigned int i = 0; i < w_dim1; i++)
		{
			b_stream << network[k].bias(i) << std::endl;
			for (unsigned int j = 0; j < w_dim2; j++)
				w_stream << network[k].weights(i, j)
					 << std::endl;
		}
	}

	// close file streams
	w_stream.close();
	b_stream.close();
}

// TODO: Create a better standard format for weights, biases files

/* Loads csv files with weights and biases to update the same on the network
 *
 * Note : Updates the weights and biases on the network
 */
void hdrnn::load_hdrnn(std::string w_file, std::string b_file)
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
void hdrnn::infer_pgm_image_from_path(std::string i_file)
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

	VectorXf image = VectorXf(INPUT_LAYER_SIZE);
	float x;

	// seek to ignore PGM boilerplate
	i_stream.seekg(13);

	// read the image file into hdrnn image
	for (unsigned int i = 0; i < INPUT_LAYER_SIZE; i++)
	{
		i_stream >> x;
		image[i] = x / 255;
	}

	std::cout << predict(image) << std::endl;
}

#endif // HDRNN_HDRNN_H
