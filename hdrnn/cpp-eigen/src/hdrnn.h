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
#include <vector>
#include <random>
#include <gflags/gflags.h>

#include "Eigen/Core"

// TODO: ADD PCG based PRNG
// #include "pcg_random.hpp"

DECLARE_bool(quiet);

#include "mnist.h"

using Eigen::Index;
using Eigen::Matrix;
using Eigen::MatrixXf;

const unsigned char MAGIC = 7;

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
	hdrnn () : network({}), epochs(0), batch_size(0), learning_rate(0) {} ;

	/* HDRNN constructor */
	hdrnn (std::vector<unsigned int>, unsigned int, unsigned int, float);

	/* HDRNN destructor */
	~hdrnn ();

	/* Train the HDR neural network */
	void train_hdrnn(std::string);

	/* Evaluate the accuracy of the hdrnn */
	void evaluate_hdrnn(unsigned int);

	/* Dump net description file */
	void dump_hdrnn(std::string) const;

	/* Load net description file */
	void load_hdrnn(std::string);

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
			weights = MatrixXf::Zero(c_dim, p_dim);
			bias = VectorXf::Zero(c_dim);
		}

		/* Update the weights and bias using the nabla
		 *
		 * @param n - Nablas with which to update
		 */
		void update(nabla& n, float factor)
		{
			weights -= (factor * n.weights);
			bias -= (factor * n.bias);
		}
	};

	/* HDRNN shape
	 *
	 * shape - specifies the number and dimension of the hidden layers to initialize
	 */
	void create_network(std::vector<unsigned int> shape)
	{
		// size of image in row vector form
		unsigned int previous_dim = mnist_loader::IMAGE_SIZE;

		// iterate over the dimensions required in the hidden layer
		for (auto dim : shape)
		{
			network.push_back(layer(dim, previous_dim));
			previous_dim = dim;
		}

		// finally, add the last layer
		network.push_back(layer(mnist_loader::DIGITS, previous_dim));
	}

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
		// pcg_extras::seed_seq_from<std::random_device> seed_source;
		// pcg32 rng(seed_source);

		std::random_device rd;
		std::default_random_engine e(rd());
		std::normal_distribution<> normal_dist(0, 1);

		for (std::size_t i = 0; i < network.size(); i++)
		{
			for (auto j = 0; j < network[i].bias.rows(); j++)
				network[i].bias[j] = normal_dist(e);
			for (auto j = 0; j < network[i].weights.rows(); j++)
				for (auto k = 0; k < network[i]
					     .weights.cols(); k++)
					network[i].weights(j,k)=normal_dist(e);
		}
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
			     ; i += batch_size)
		{
			// Perform Backpropagation on the batch
			for (std::size_t j = 0; j < batch_size; j++)
				back_propogate(nablas,
					       mnist_loader::train[i+j].data,
					       mnist_loader::train[i+j].label);

			// Update the weights and biases of the network
			for (std::size_t j = 0; j < network.size(); j++)
				network[j].update(nablas[j], learning_rate / batch_size);

			// Zero out the nabla matrixes
			for (std::size_t j = 0; j < nablas.size(); j++)
				nablas[j].zero_out();
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
		VectorXf y = VectorXf::Zero(mnist_loader::DIGITS);
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
	unsigned int epochs;
	unsigned int batch_size;
	float learning_rate;
};

/* HDRNN constructor
 *
 * shape - specifies the number and dimension of the hidden layers to initialize
 * e     - number of epochs to train the network
 * bs    - batch size for mSGD
 * lr    - learning rate for SGD
 */
hdrnn::hdrnn(std::vector<unsigned int> shape, unsigned int e,
	unsigned int bs, float lr)
{
	create_network(shape);

	// parameters for the learning algorithm
	epochs = e;
	batch_size = bs;
	learning_rate = lr;
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
	if (!mnist_loader::read_mnist(dataset))
	{
		std::cerr << "could not load the dataset" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	generate_random_weights();

	// TODO: Generate PCG-based weights and bias initializations
	for (unsigned int i = 0; i < epochs; i++)
	{
		std::random_device r;
		std::default_random_engine e1(r());
		std::shuffle(
		     std::begin(mnist_loader::train),
		     std::end(mnist_loader::train), e1
		     );
		mini_batch_sgd();
		if (!FLAGS_quiet)
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

/* Dumps neural network description file (see hdrnn/README.md)
 *
 * Note : Creates .nn file with the weights and biases of the network
 */
void hdrnn::dump_hdrnn(std::string n_file) const
{
	// File output stream for .nn file
	std::ofstream n_stream;

	// Open the .nn file
	n_stream.open(n_file, std::ios::out | std::ios::binary);
	if (!n_stream)
	{
		std::cerr << "Could not open" << n_file << std::endl;
		std::exit(EXIT_FAILURE);
	}

	unsigned char hidden_layers = MAGIC;

	// write the MAGIC byte
	n_stream.write(reinterpret_cast<char*>(&hidden_layers), 1);

	// write the number of hidden layers
	// TODO: assert this type bound somewhere
	hidden_layers = network.size() - 1;
	n_stream.write(reinterpret_cast<char*>(&hidden_layers), 1);

	// write the hidden layer sizes
	for (auto l : network)
	{
		unsigned char size = l.bias.size();
		n_stream.write(reinterpret_cast<char *>(&size), 1);
	}

	// write the biases
	for (auto l : network)
	{
		n_stream.write(
			reinterpret_cast<char *>(l.bias.data()),
			l.bias.size() * sizeof(float)
			);
	}

	// write the weights
	for (auto l : network)
	{
		n_stream.write(
			reinterpret_cast<char *>(l.weights.data()),
			l.weights.size() * sizeof(float)
			);
	}

	// close file stream
	n_stream.close();
}

/* Loads neural network description file (see hdrnn/README.md)
 *
 * Note : Updates the weights and biases on the network
 */
void hdrnn::load_hdrnn(std::string n_file)
{
	// File input stream for .nn file
	std::ifstream n_stream;

	// Open the .nn file
	n_stream.open(n_file, std::ios::in | std::ios::binary);
	if (!n_stream)
	{
		std::cerr << "Could not open" << n_file << std::endl;
		std::exit(EXIT_FAILURE);
	}

	unsigned char hidden_layers;

	// read the MAGIC byte
	n_stream.read(reinterpret_cast<char *>(&hidden_layers), 1);

	if (hidden_layers != MAGIC)
	{
		std::cerr << "could not verify magic value: "
			<< hidden_layers
			<< std::endl;
		std::exit(EXIT_FAILURE);
	}

	// read the size
	n_stream.read(reinterpret_cast<char *>(&hidden_layers), 1);

	// read the shape
	std::vector<unsigned int> shape;
	unsigned int size;

	for (unsigned char i = 0; i < hidden_layers; i++)
	{
		size = 0;
		n_stream.read(reinterpret_cast<char *>(&size), 1);
		shape.push_back(size);
	}

	// create the network
	create_network(shape);

	// read the biases
	for (auto l : network)
	{
		n_stream.read(
			reinterpret_cast<char *>(l.bias.data()),
			l.bias.size() * sizeof(float)
			);
	}

	// read the weights
	for (auto l : network)
	{
		n_stream.read(
			reinterpret_cast<char *>(l.weights.data()),
			l.weights.size() * sizeof(float)
			);
	}

	// close file stream
	n_stream.close();
}

// TODO: Change to a more tradition PGM read approach
/* Infer the digit in an image from path using the network
 *
 * Note : Changes the current image
 *        Outputs the digit that the HDRNN thinks the image represents
 * i_file - filename of a PGM image (in a particular format)
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

	VectorXf image = VectorXf(mnist_loader::IMAGE_SIZE);
	float x;

	// seek to ignore PGM boilerplate
	i_stream.seekg(13);

	// read the image file into hdrnn image
	for (unsigned int i = 0; i < mnist_loader::IMAGE_SIZE; i++)
	{
		i_stream >> x;
		image[i] = x / 255;
	}

	std::cout << predict(image) << std::endl;
}

#endif // HDRNN_HDRNN_H
