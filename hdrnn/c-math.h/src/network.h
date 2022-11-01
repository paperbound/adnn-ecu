/*
 * @author Prasanth Thomas Shaji
 *
 * HDR Neural Network
 *
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <math.h>
#include <string.h>
#include <time.h>

#include "images.h"

/* Hyperparameters */

int epochs = 30;
int batchSize = 10;
double eta = 3;

#define INPUT_LAYER_SIZE 784
#define OUTPUT_LAYER_SIZE 10

// Using just the one hidden layer
#define HIDDEN_LAYER_SIZE 30

/* MNIST Dataset */

extern double train_images[NUM_TRAIN][SIZE];
extern double test_images[NUM_TEST][SIZE];
extern int train_labels[NUM_TRAIN];
extern int test_labels[NUM_TEST];

extern int info_image[LEN_INFO_IMAGE];
extern int info_label[LEN_INFO_LABEL];

/* Current Image */

extern double image[SIZE]; // input layer / loaded image

/* HDR Neural Network */

typedef struct
{
	double bias;
	double *weights;
	double *nabla_w;
} Neuron;

typedef struct LayerT
{
	int size;
	int incidents;
	Neuron *neurons;
	double *activations;
	double *z_values;
	double *nabla_b;
	struct LayerT *next;
	struct LayerT *previous;
} Layer;

typedef struct
{
	Layer *layers;
	int depth;
} Network;

static double randn(double, double);

static double sigmoid(double);
static double sigmoid_prime(double);

static int get_integer(FILE *);
static double get_double(FILE *);

static int prediction(Network *);
static void test_network(Network *);

static void add_layer(Network *, int, int);

static void mini_batch_sgd(Network *);
static void zero_nablas(Network *);

static void feed_forward(Network *, double *);
static void back_propogate(Network *, double *);

/* HDRNN Neural Network
 *
 * Input Layer -> Hidden Layer -> Output Layer
 *  (image)
 */
void initHDRNN(Network *network)
{
	// Network Initialisations
	// Start with no first layer
	// feedforward() assumes image as first set of activations
	network->depth = 3;

	add_layer(network, HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE);
	add_layer(network, OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
}

static double randn (double mu, double sigma)
{
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;

	if (call == 1)
		{
		call = !call;
		return (mu + sigma * (double) X2);
		}

	do
	{
		U1 = -1 + ((double) rand () / RAND_MAX) * 2;
		U2 = -1 + ((double) rand () / RAND_MAX) * 2;
		W = pow (U1, 2) + pow (U2, 2);
	}
	while (W >= 1 || W == 0);

	mult = sqrt ((-2 * log (W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double) X1);
}

static void add_layer(Network *network, int size, int incidents)
{
	Layer *layer = (Layer *)calloc(1, sizeof(Layer));
	layer->incidents = incidents;
	layer->size = size;
	layer->activations = (double *)calloc(size, sizeof(double));
	layer->z_values = (double *)calloc(size, sizeof(double));
	layer->nabla_b = (double *)calloc(size, sizeof(double));
	layer->next = NULL;
	layer->previous = NULL;

	Neuron *neurons = (Neuron *)calloc(size, sizeof(Neuron));
	for (int i = 0; i < size; i++)
	{
		Neuron *neuron = neurons + i;
		double r = sqrt(2 / (double)size);
		neuron->bias = r * randn(0, 1);
		double *weights = (double *)calloc(incidents, sizeof(double));
		for (int j = 0; j < incidents; j++)
			weights[j] = r * randn(0, 1);
		neuron->weights = weights;
		neuron->nabla_w = (double *)calloc(incidents, sizeof(double));
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

/* TODO: Fix the ugly getline stuff later */
static int get_integer(FILE *fd)
{
	// getline
	int rows;
	size_t llen = 21;
	char *line = (char *)calloc(llen, sizeof(char));

	if (getline(&line, &llen, fd) == -1)
	{
		fprintf(stderr, "Couldn't read Rows\n");
		exit(-1);
	}
	if (sscanf(line, "%d", &rows) == EOF)
	{
		fprintf(stderr, "Couldn't parse Rows\n");
		exit(1);
	}
	free(line);
	return rows;
}

/* TODO: Fix the ugly getline stuff later */
static double get_double(FILE *fd)
{
	// getline
	double value;
	size_t llen = 21;
	char *line = (char *)calloc(llen, sizeof(char));

	if (getline(&line, &llen, fd) == -1)
	{
		fprintf(stderr, "Couldn't read Bias");
		exit(-1);
	}
	if (sscanf(line, "%lf", &value) == EOF)
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
			fprintf(stderr, "sizes dont match\n");
			exit(-1);
		}
		for (int i = 0; i < rows; i++)
			layer->neurons[i].bias = get_double(fd);
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
				layer->neurons[i].weights[j] = get_double(fd);
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

static double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_prime(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

void trainHDRNN(Network *network)
{
	srand(time(NULL));
	for (int i = 0; i < epochs; i++)
	{
		shuffle_train_indexes();
		mini_batch_sgd(network);
		test_network(network);
	}
}

/* Stochastic Gradient Descend */

static void mini_batch_sgd(Network *network)
{
	// Initialisations
	double factor = eta / batchSize;
	double **nabla_b = calloc(network->depth - 1, sizeof(double *));
	double ***nabla_w = calloc(network->depth - 1, sizeof(double **));

	int k = 0;
	Layer *layer = network->layers;
	while (layer != NULL & layer->next != NULL)
	{
		nabla_b[k] = (double *)calloc(layer->size, sizeof(double));
		nabla_w[k] = (double **)calloc(layer->size, sizeof(double *));
		for (int l = 0; l < layer->size; l++)
		{
			nabla_w[k][l] = (double *)calloc(layer->incidents, sizeof(double));
		}
		k += 1;
		layer = layer->next;
	}

	for (int i = 0; i < NUM_TRAIN; i += batchSize)
	{
		// Set Batch Nablas to zero
		k = 0;
		layer = network->layers;
		while (layer != NULL & layer->next != NULL)
		{
			for (int l = 0; l < layer->size; l++)
			{
				nabla_b[k][l] = 0;
				for (int m = 0; m < layer->incidents; m++)
					nabla_w[k][l][m] = 0;
			}
			k += 1;
			layer = layer->next;
		}

		for (int j = 0; j < batchSize; j++)
		{
			zero_nablas(network);
			double label[OUTPUT_LAYER_SIZE] = {0};
			// update_mini_batch
			load_train_image(i + j);
			label[get_train_label(i + j)] = 1;
			back_propogate(network, label);

			// Accumulate Batch Nablas
			k = 0;
			layer = network->layers;
			while (layer != NULL & layer->next != NULL)
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
			{
				neuron->weights[j] -= (factor * nabla_w[k][i][j]);
			}
		}
		layer = layer->next;
	}

}

static void zero_nablas(Network *network)
{
	Layer *layer = network->layers;

	while (layer != NULL)
	{
		memset(layer->nabla_b, 0, layer->size * sizeof(double));
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			memset(neuron->nabla_w, 0, layer->incidents * sizeof(double));
		}
		layer = layer->next;
	}
}

/* Evaluate Network */

static int prediction(Network *network)
{
	int prediction = -1;
	double highest = 0;

	Layer *output_layer = network->layers;
	while (output_layer != NULL && output_layer->next != NULL)
		output_layer = output_layer->next;

	for (int i = 0; i < OUTPUT_LAYER_SIZE; i++)
	{
		if (highest < output_layer->activations[i])
		{
			highest = output_layer->activations[i];
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
		load_image(test_images, i);
		feed_forward(network, image);
		guess = prediction(network);
		correct += (guess == test_labels[i]);
		// printf("guess : %d label %d\n", guess, test_labels[i]);
	}
	printf("Network has classified %d (%d) images correctly\n", correct, NUM_TEST);
}

/* Forward Propogation
 *
 * TODO: Have a better strategy than allocating and deallocating memory here
 */
static void feed_forward(Network *network, double *image)
{
	// First layer is the image
	double *activations = image;

	Layer *layer = network->layers;
	while (layer != NULL)
	{
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			double zvalue = 0;
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

static void back_propogate(Network *network, double *y)
{
	feed_forward(network, image);

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
		last_layer->nabla_b[i] = (last_layer->activations[i] - y[i]) *
			sigmoid_prime(last_layer->z_values[i]);

		Neuron *neuron = &last_layer->neurons[i];
		for (int j = 0; j < second_last_layer->size; j++)
		{
			neuron->nabla_w[j] = last_layer->nabla_b[i] *
				second_last_layer->activations[j];
		}
	}

	// Iterate over layers in reverse order from the last one
	layer = second_last_layer;
	while (layer != NULL)
	{
		Layer *earlier_layer = layer->next;
		for (int i = 0; i < layer->size; i++)
		{
			Neuron *neuron = &layer->neurons[i];
			double sp = sigmoid_prime(layer->z_values[i]);
			// Iterate over earlier layer
			for (int j = 0; j < earlier_layer->size; j++)
			{
				double delta = earlier_layer->nabla_b[j];
				Neuron *e_neuron = &earlier_layer->neurons[j];
				layer->nabla_b[i] += (delta * e_neuron->weights[i]);
			}
			layer->nabla_b[i] *= sp;

			double *activations;
			if (layer->previous == NULL)
				activations = image;
			else
				activations = layer->previous->activations;

			for (int j = 0; j < layer->incidents; j++)
			{
				neuron->nabla_w[j] = (layer->nabla_b[i] *
					activations[j]);
			}
		}
		layer = layer->previous;
	}
}

/* Dump csv file of weights and biases
 */
void dumpWeights(Network *network, char *wfile, char *bfile)
{
	FILE *fd;

	// bias

	if ((fd = fopen(bfile, "w")) == NULL)
	{
		fprintf(stderr, "couldn't open bias file %s\n", bfile);
		exit(-1);
	}

	Layer *layer = network->layers;
	while (layer != NULL)
	{
		fprintf(fd, "%d\n", layer->size);
		for (int i = 0; i < layer->size; i++)
			fprintf(fd, "%f\n", layer->neurons[i].bias);
		layer = layer->next;
	}

	fclose(fd);

	// weights

	if ((fd = fopen(wfile, "w")) == NULL)
	{
		fprintf(stderr, "couldn't open weight file\n");
		exit(-1);
	}

	layer = network->layers;
	while (layer != NULL)
	{
		fprintf(fd, "%d\n", layer->size);
		fprintf(fd, "%d\n", layer->incidents);
		for (int i = 0; i < layer->size; i++)
			for (int j = 0; j < layer->incidents; j++)
				fprintf(fd, "%f\n", layer->neurons[i].weights[j]);
		layer = layer->next;
	}

	fclose(fd);
}

#endif
