/*
 * @author Prasanth Thomas Shaji
 *
 * HDR Neural Network
 *
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include "images.h"

/* Hyperparameters */

int epochs = 20;
int batchSize = 10;
float eta = 3.0;

#define INPUT_LAYER_SIZE 784
#define OUTPUT_LAYER_SIZE 10

// Using just the one hidden layer
#define HIDDEN_LAYER_SIZE 30

/* MNIST Dataset */

extern float train_images[NUM_TRAIN][SIZE];
extern float test_images[NUM_TEST][SIZE];
extern int train_labels[NUM_TRAIN];
extern int test_labels[NUM_TEST];

extern int info_image[LEN_INFO_IMAGE];
extern int info_label[LEN_INFO_LABEL];

/* Current Image */

extern float image[SIZE]; // input layer / loaded image

/* HDR Neural Network */

typedef struct
{
	float bias;
	float zvalue;
	float output;
	float *weights;
} Neuron;

typedef struct LayerT
{
	int size;
	int incidents;
	Neuron *neurons;
	struct LayerT *next;
} Layer;

typedef struct
{
	Layer *layers;
} Network;

static float sigmoid(float);

static int get_integer(FILE *);
static float get_float(FILE *);
static int prediction(Network *);

static void add_layer(Network *, int, int);
static void mini_batch_sgd(Network *);
static void test_network(Network *);
static void feed_forward(Network *, float *);
static void back_propogate(Network *);

/* HDRNN Neural Network
 *
 * Input Layer -> Hidden Layer -> Output Layer
 *  (image)
 */
void initHDRNN(Network *network)
{
	add_layer(network, HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE);
	add_layer(network, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
}

static void add_layer(Network *network, int size, int incidents)
{
	Layer *layer = (Layer *)malloc(sizeof(Layer));
	layer->incidents = incidents;
	layer->size = size;

	Neuron *neurons = (Neuron *)malloc(size * sizeof(Neuron));
	for (int i = 0; i < size; i++)
	{
		Neuron *neuron = neurons + i;
		neuron->bias = rand() / (float)RAND_MAX;
		float *weights = (float *)malloc(incidents * sizeof(float));
		for (int j = 0; j < incidents; j++)
			weights[j] = 0.7 * (rand() / (float)RAND_MAX);
		neuron->weights = weights;
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
	}
}

void trainHDRNN(Network *network)
{
	for (int i = 0; i < epochs; i++)
	{
		// TODO: Yuan does random shuffle here
		mini_batch_sgd(network);
		test_network(network);
	}
}

/* TODO: Fix the ugly getline stuff later */
static int get_integer(FILE *fd)
{
	// getline
	int rows;
	size_t llen = 21;
	char *line = (char *)malloc(llen * sizeof(char));

	if (getline(&line, &llen, fd) == -1)
	{
		fprintf(stderr, "Couldn't read Rows");
		exit(-1);
	}
	if (sscanf(line, "%d", &rows) == EOF)
	{
		fprintf(stderr, "Couldn't parse Rows");
		exit(1);
	}
	free(line);
	return rows;
}

/* TODO: Fix the ugly getline stuff later */
static float get_float(FILE *fd)
{
	// getline
	float value;
	size_t llen = 21;
	char *line = (char *)malloc(llen * sizeof(char));

	if (getline(&line, &llen, fd) == -1)
	{
		fprintf(stderr, "Couldn't read Bias");
		exit(-1);
	}
	if (sscanf(line, "%f", &value) == EOF)
	{
		fprintf(stderr, "Couldn't parse Rows");
		exit(1);
	}
	free(line);
	return value;
}

/* Load csv file of weights into the Network
 * weights into network
 */
void loadHDRNN(Network *network, char *wfile, char *bfile)
{
	FILE *fd;
	int rows, cols;

	// bias

	if ((fd = fopen(bfile, "r")) == NULL)
	{
		fprintf(stderr, "couldn't open bias file\n");
		exit(-1);
	}

	Layer *layer = network->layers;
	while (layer != NULL)
	{
		rows = get_integer(fd);
		if (rows != layer->size)
		{
			fprintf(stderr, "sizes dont match");
			exit(-1);
		}
		for (int i = 0; i < rows; i++)
			layer->neurons[i].bias = get_float(fd);
		layer = layer->next;
	}

	fclose(fd);

	// weights

	if ((fd = fopen(wfile, "r")) == NULL)
	{
		fprintf(stderr, "couldn't open weight file\n");
		exit(-1);
	}

	layer = network->layers;
	while (layer != NULL)
	{
		rows = get_integer(fd);
		cols = get_integer(fd);
		if (rows != layer->size || cols != layer->incidents)
		{
			fprintf(stderr, "sizes dont match");
			exit(-1);
		}
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				layer->neurons[i].weights[j] = get_float(fd);
		layer = layer->next;
	}

	fclose(fd);
}

void inferImage(Network *network, char *wfile)
{
	load_infer_image(wfile);
	feed_forward(network, image);
	int pred = prediction(network);
	for (int i = 0; i < SIZE; i++)
	{
		if (image[i] != 0)
		{
			printf("x ");
		} else
		{
			printf("  ");
		}
		if ( (i+1) % 28 == 0)
			printf("\n");
	}
	printf("Network predicts %d\n", pred);
}

/* Sigmoid Activation Function */

static float sigmoid(float x)
{
	return 1.0 / (1.0 + exp(-x));
}

/* static float sigmoid_prime(float x) */
/* { */
/* 	return sigmoid(x) * (1 - sigmoid(x)); */
/* } */

/* Stochastic Gradient Descend */

static void mini_batch_sgd(Network *network)
{
	for (int i = 0; i < NUM_TRAIN; i += batchSize)
	{
		for (int j = 0; j < batchSize; j++)
		{
			// update_mini_batch
			load_image(train_images, i + j);
			back_propogate(network);
		}
	}
}

/* Evaluate Network */

static int prediction(Network *network)
{
	int prediction = -1;
	float highest = 0;
	for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
	{
		Layer *output = network->layers;
		while (output != NULL && output->next != NULL)
			output = output->next;
		if (highest < output->neurons[i].output)
		{
			highest = output->neurons[i].output;
			prediction = i;
		}
	}
	return prediction;
}

static void test_network(Network *network)
{
	int correct = 0, guess;
	for (int i = 0; i < NUM_TEST; i++)
	{
		feed_forward(network, image);
		guess = prediction(network);
		correct += guess == test_labels[i];
	}
	printf("Network has classified %d (%d) images correctly", correct, NUM_TEST);
}

/* Forward Propogation
 *
 * TODO: Have a better strategy than allocating and deallocating memory here
 */
static void feed_forward(Network *network, float *image)
{
	Layer *layer = network->layers;
	float *activations = image;
	while (layer != NULL)
	{
		float *nextActivations = (float *)malloc(layer->size * sizeof(float));
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			float zvalue = 0;
			for (int i = 0; i < layer->incidents; i++)
			{
				zvalue += neuron->weights[i] * activations[i];
			}
			neuron->zvalue = zvalue + neuron->bias;
			neuron->output = sigmoid(neuron->zvalue);
			nextActivations[i] = neuron->output;
		}
		layer = layer->next;
		if (activations != image)
		{
			free(activations);
		}
		activations = nextActivations;
	}
	// free the last layer of activations
	free(activations);
}

/* Back Propogation */

static void back_propogate(Network *network)
{
	/* feed_forward(network, image); */
	/* float factor = eta / batchSize; */
	/* // Take z values and activations of the last layer */
	/* Layer *layer = network->layers; */
	/* while (layer != NULL && layer->next != NULL) */
	/* { */
	/* 	layer = layer->next; */
	/* } */
	/* float *zvalues = (float *)malloc(layer->size * sizeof(float)); */
	/* float *outputs = (float *)malloc(layer->size * sizeof(float)); */
}

#endif
