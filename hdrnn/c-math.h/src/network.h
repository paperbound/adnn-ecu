/*
 * @author Prasanth Thomas Shaji
 *
 * HDR Neural Network
 *
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include <string.h> // for memset
#include <stdbool.h>

#include "images.h"
#include "random.h"

/* Output the network */
bool quiet = false;
char *nfile = "c-math.nn";
const unsigned char MAGIC = 7;

/* Image file */
extern char *ifile;

/* Default network */
unsigned int depth = 3;
unsigned int default_shape[] = {784, 32, 10};
unsigned int *shape = default_shape;

/* (Default) Hyperparameters */
unsigned int epochs = 30;
unsigned int batchSize = 10;
float eta = 3.0f;

/* MNIST Dataset */
extern float train_images[NUM_TRAIN][SIZE];
extern float test_images[NUM_TEST][SIZE];
extern int train_labels[NUM_TRAIN];
extern int test_labels[NUM_TEST];

extern int info_image[LEN_INFO_IMAGE];
extern int info_label[LEN_INFO_LABEL];

/* Current Image (Input Layer or Loaded Image)*/
extern float image[SIZE];

/* HDR Neural Network */
typedef struct
{
	float bias;
	float *weights;
	float *nabla_w;
} Neuron;

typedef struct LayerT
{
	int size;
	int incidents;
	Neuron *neurons;
	float *activations;
	float *z_values;
	float *nabla_b;
	struct LayerT *next;
	struct LayerT *previous;
} Layer; // Network layers except for input

typedef struct
{
	Layer *layers;
	int depth;
} Network; // HDRNN

static float sigmoid(float);
static float sigmoid_prime(float);

static int  prediction(Network *);
static void generate_random_weights(Network *);

static void add_layer(Network *, int, int);

static void mini_batch_sgd(Network *);
static void zero_nablas(Network *);

static void feed_forward(Network *, float *);
static void back_propogate(Network *, float *);

/* HDRNN Neural Network
 *
 * Input Layer -> Hidden Layer(s) -> Output Layer
 */
void initHDRNN(Network *network)
{
	// Network Initialisations
	// * Start with no first layer
	// * feedforward() assumes _image_ as first set of activations

	unsigned int incidents, size;
	incidents = SIZE; // First layer is 784 (image size)

	for (size_t i = 1; i < depth; i++)
	{
		size = shape[i];
		add_layer(network, size, incidents);
		incidents = size;
	}
}

static void add_layer(Network *network, int size, int incidents)
{
	Layer *layer = (Layer *)calloc(1, sizeof(Layer));
	layer->incidents = incidents;
	layer->size = size;
	layer->activations = (float *)calloc(size, sizeof(float));
	layer->z_values = (float *)calloc(size, sizeof(float));
	layer->nabla_b = (float *)calloc(size, sizeof(float));
	layer->next = NULL;
	layer->previous = NULL;

	Neuron *neurons = (Neuron *)calloc(size, sizeof(Neuron));
	for (int i = 0; i < size; i++)
	{
		Neuron *neuron = neurons + i;
		float *weights = (float *)calloc(incidents, sizeof(float));
		neuron->weights = weights;
		neuron->nabla_w = (float *)calloc(incidents, sizeof(float));
	}
	layer->neurons = neurons;

	Layer *last = network->layers;
	if (last == NULL)
	{
		network->layers = layer;
	}
	else
	{
		while (last->next != NULL)
			last = last->next;
		last->next = layer;
		layer->previous = last;
	}
}

static void generate_random_weights(Network *network)
{
	// Setup Random Number Generator
	bitgen_t bitgen;

	bitgen = init_prng();

	Layer *layer = network->layers;
	while (layer != NULL)
	{
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			neuron->bias = random_standard_normal_f(&bitgen);
			for (int j = 0; j < layer->incidents; j++)
				neuron->weights[j] =  random_standard_normal_f(&bitgen);
		}
		layer = layer->next;
	}
}

/* Load net description file into the network
 */
void loadHDRNN(Network *network)
{
	FILE *fd;
	unsigned int size;
	size_t ret_code;

	if ((fd = fopen(nfile, "rb")) == NULL)
	{
		fprintf(stderr, "couldn't open net file\n");
		exit(-1);
	}

	ret_code = fread(&size, 1, 1, fd);
	if (ret_code != 1)
	{
		fprintf(stderr, "error reading net file, magic\n");
		exit(-1);
	}

	// size should have magic value
	if (size != MAGIC)
	{
		fprintf(stderr, "could not verify magic value, found %d\n", size);
		exit(-1);
	}

	size = 0;
	ret_code = fread(&size, 1, 1, fd);
	if (ret_code != 1)
	{
		fprintf(stderr, "error reading net file, size\n");
		exit(-1);
	}

	// Get the shape
	depth = size + 2;
	shape = (unsigned int *) calloc(depth, sizeof(int));
	unsigned int value, index;

	shape[0] = SIZE; // input layer size is 784
	index = 1;

	// Read in shape
	for (unsigned int i = 0; i < size; i++)
	{
		value = 0;
		ret_code = fread(&value, 1, 1, fd);
		if (ret_code != 1)
		{
			fprintf(stderr, "error reading net file, shape\n");
			exit(-1);
		}
		shape[index] = value;
		index += 1;
	}

	shape[index] = DIGITS; // output layer size

	initHDRNN(network);

	// Read Biases
	size_t lsize;
	Layer *layer = network->layers; // first layer

	for (size_t i = 1; i < depth; i++)
	{
		lsize = shape[i];
		for (size_t j = 0; j < lsize; j++)
		{
			ret_code = fread(&(layer->neurons[j].bias), sizeof(float), 1, fd);
			if (ret_code != 1)
			{
				fprintf(stderr, "error reading net file, bias\n");
				exit(-1);
			}
		}
		layer = layer->next;
	}

	// Read Weights
	layer = network->layers; // go back to first layer

	size_t incidents;
	incidents = SIZE; // First layer is 784 (image size)

	for (size_t i = 1; i < depth; i++)
	{
		lsize = shape[i];
		for (size_t j = 0; j < lsize; j++)
		{
			ret_code = fread(layer->neurons[j].weights, sizeof(float), incidents, fd);
			if (ret_code != incidents)
			{
				fprintf(stderr, "error reading net file, weights\n");
				exit(-1);
			}
		}
		layer = layer->next;
		incidents = lsize;
	}

	fclose(fd);
}

void inferImage(Network *network)
{
	load_infer_image(ifile);
	feed_forward(network, image);
	unsigned short pred = prediction(network);
	for (unsigned short i = 0; i < SIZE; i++)
	{
		if (image[i] != 0)
		{
			printf("x ");
		} else
		{
			printf("  ");
		}
		if ( (i+1) % DIMENSION == 0)
			printf("\n");
	}
	printf("Network predicts %d\n", pred);
}

/* Sigmoid Activation Function */

static float sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

static float sigmoid_prime(float x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

void test_network(Network *network)
{
	int correct = 0, guess;
	for (int i = 0; i < NUM_TEST; i++)
	{
		load_image(test_images, i);
		feed_forward(network, image);
		guess = prediction(network);
		correct += (guess == test_labels[i]);
		// printf("guess : %d label %d\n", guess, test_labels[i]);
	}
	printf("Network classified %d (%d) images correctly\n", correct, NUM_TEST);
}

void trainHDRNN(Network *network, bool quiet)
{
	load_mnist();
	generate_random_weights(network);
	for (int i = 0; i < epochs; i++)
	{
		mini_batch_sgd(network);
		if (!quiet)
			test_network(network);
	}
}

/* Stochastic Gradient Descend */

static void mini_batch_sgd(Network *network)
{
	// Batch Allocations
	float factor = eta / batchSize;
	float **nabla_b = calloc(depth - 1, sizeof(float *));
	float ***nabla_w = calloc(depth - 1, sizeof(float **));
	float label[DIGITS] = {0};

	int k = 0;
	Layer *layer = network->layers;
	while (layer != NULL)
	{
		nabla_b[k] = (float *)calloc(layer->size, sizeof(float));
		nabla_w[k] = (float **)calloc(layer->size, sizeof(float *));
		for (int l = 0; l < layer->size; l++)
			nabla_w[k][l] = (float *)calloc(layer->incidents, sizeof(float));
		k += 1;
		layer = layer->next;
	}

	for (int i = 0; i < NUM_TRAIN; i += batchSize)
	{
		// Set Batch Nablas to zero
		k = 0;
		layer = network->layers;
		while (layer != NULL)
		{
			memset(nabla_b[k], 0, layer->size * sizeof(float));
			for (int l = 0; l < layer->size; l++)
				memset(nabla_w[k][l], 0, layer->incidents * sizeof(float));
			k += 1;
			layer = layer->next;
		}

		for (int j = 0; j < batchSize; j++)
		{
			// Update the mini batch
			zero_nablas(network);

			// Load training image and label
			load_train_image(i + j);
			memset(label, 0, DIGITS * sizeof(float));
			label[get_train_label(i + j)] = 1;

			// Back propagation
			feed_forward(network, image);
			back_propogate(network, label);

			// Accumulate Batch Nablas
			k = 0;
			layer = network->layers;
			while (layer != NULL)
			{
				for (int l = 0; l < layer->size; l++)
				{
					nabla_b[k][l] += layer->nabla_b[l];
					Neuron *neuron = &layer->neurons[l];
					for (int m = 0; m < layer->incidents; m++)
						nabla_w[k][l][m] += neuron->nabla_w[m];
				}
				k += 1;
				layer = layer->next;
			}
		}

		// Update with Batch Nablas
		k = 0;
		layer = network->layers;
		while (layer != NULL)
		{
			for (int i = 0; i < layer->size; i++)
			{
				Neuron *neuron = &layer->neurons[i];
				neuron->bias -= (factor * nabla_b[k][i]);
				for (int j = 0; j < layer->incidents; j++)
					neuron->weights[j] -= (factor * nabla_w[k][i][j]);
			}
			k += 1;
			layer = layer->next;
		}
	}

	// Batch Deallocations
	k = 0;
	layer = network->layers;
	while (layer != NULL)
	{
		free(nabla_b[k]);
		for (int l = 0; l < layer->size; l++)
			free(nabla_w[k][l]);
		free(nabla_w[k]);
		k += 1;
		layer = layer->next;
	}

	free(nabla_b);
	free(nabla_w);
}

static void zero_nablas(Network *network)
{
	Layer *layer = network->layers;

	while (layer != NULL)
	{
		memset(layer->nabla_b, 0, layer->size * sizeof(float));
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			memset(neuron->nabla_w, 0, layer->incidents * sizeof(float));
		}
		layer = layer->next;
	}
}

/* Evaluate Network */

static int prediction(Network *network)
{
	int prediction = -1;
	float highest = 0;

	Layer *output_layer = network->layers;
	while (output_layer != NULL && output_layer->next != NULL)
		output_layer = output_layer->next;

	for (int i = 0; i < DIGITS; i++)
	{
		if (highest < output_layer->activations[i])
		{
			highest = output_layer->activations[i];
			prediction = i;
		}
	}
	return prediction;
}

/* Forward Propogation
 *
 * TODO: Have a better strategy than allocating and deallocating memory here
 */
static void feed_forward(Network *network, float *image)
{
	// First layer is the image
	float *activations = image;

	Layer *layer = network->layers;
	while (layer != NULL)
	{
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			float zvalue = 0;
			for (int j = 0; j < layer->incidents; j++)
			{
				zvalue += neuron->weights[j] * activations[j];
			}
			layer->z_values[i] = zvalue + neuron->bias;
			layer->activations[i] = sigmoid(layer->z_values[i]);
		}
		activations = layer->activations;
		layer = layer->next;
	}
}

/* Back Propogation */

static void back_propogate(Network *network, float *label)
{
	// Take last and second last layers
	Layer *layer = network->layers;
	while (layer != NULL && layer->next != NULL)
	{
		layer = layer->next;
	}
	Layer *last_layer = layer, *second_last_layer = layer->previous;

	if (last_layer == NULL || second_last_layer == NULL)
		return;

	for (int i = 0; i < last_layer->size; i++)
	{
		last_layer->nabla_b[i] = (last_layer->activations[i] - label[i]) *
			sigmoid_prime(last_layer->z_values[i]);

		Neuron *neuron = &last_layer->neurons[i];
		for (int j = 0; j < second_last_layer->size; j++)
		{
			neuron->nabla_w[j] = last_layer->nabla_b[i] *
				second_last_layer->activations[j];
		}
	}

	// Iterate over layers in _reverse_ order from the last one
	layer = second_last_layer;
	float *activations;
	while (layer != NULL)
	{
		Layer *earlier_layer = layer->next;
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			float sp = sigmoid_prime(layer->z_values[i]);
			// Iterate over earlier layer
			for (int j = 0; j < earlier_layer->size; j++)
			{
				float delta = earlier_layer->nabla_b[j];
				Neuron *e_neuron = &earlier_layer->neurons[j];
				layer->nabla_b[i] += (delta * e_neuron->weights[i]);
			}
			layer->nabla_b[i] = layer->nabla_b[i] * sp;

			if (layer->previous == NULL)
				activations = image;
			else
				activations = layer->previous->activations;

			for (int j = 0; j < layer->incidents; j++)
			{
				neuron->nabla_w[j] = layer->nabla_b[i] * activations[j];
			}
		}
		layer = layer->previous;
	}
}

/* Dump the network in network description file format
 */
void dumpWeights(Network *network)
{
	FILE *fd;

	if ((fd = fopen(nfile, "wb")) == NULL)
	{
		fprintf(stderr, "couldn't open net file %s to write\n", nfile);
		exit(-1);
	}

	size_t written;
	written = fwrite(&MAGIC, sizeof(unsigned char), 1, fd);
	if (written != 1)
	{
		fprintf(stderr, "could not write magic value\n");
	}

	unsigned char size = depth - 2;
	fwrite(&size, sizeof(unsigned char), 1, fd);

	// Write the shape
	unsigned char value;

	for (size_t i = 1; i < depth - 1; i++)
	{
		value = shape[i];
		fwrite(&value, sizeof(unsigned char), 1, fd);
	}

	// Write the biases
	size_t lsize;
	Layer *layer = network->layers; // first layer

	for (size_t i = 1; i < depth; i++)
	{
		lsize = shape[i];
		for (size_t j = 0; j < lsize; j++)
			fwrite(&(layer->neurons[j].bias), sizeof(float), 1, fd);
		layer = layer->next;
	}

	// Write the Weights
	layer = network->layers; // go back to first layer

	size_t incidents;
	incidents = SIZE; // First layer is 784 (image size)

	for (size_t i = 1; i < depth; i++)
	{
		lsize = shape[i];
		for (size_t j = 0; j < lsize; j++)
			fwrite(layer->neurons[j].weights, sizeof(float), incidents, fd);
		layer = layer->next;
		incidents = lsize;
	}

	fclose(fd);
}

#endif
