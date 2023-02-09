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

#include <cstdio>
#include <cstdlib>

/* Location for MNIST Data */
#define TRAIN_IMAGE "train-images-idx3-ubyte"
#define TRAIN_LABEL "train-labels-idx1-ubyte"
#define TEST_IMAGE "t10k-images-idx3-ubyte"
#define TEST_LABEL "t10k-labels-idx1-ubyte"

/* Image Parameters */
#define SIZE 784 // 28*28

#define NUM_TRAIN 50000
#define NUM_VALIDATE 10000
#define NUM_TEST 10000

#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256

/* Training and Testing Data */

int train_indexer[NUM_TRAIN];

float train_images[NUM_TRAIN][SIZE];
float validate_images[NUM_VALIDATE][SIZE];
float test_images[NUM_TEST][SIZE];

int train_labels[NUM_TRAIN];
int validate_labels[NUM_VALIDATE];
int test_labels[NUM_TEST];

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

/* Current Image */

float image[SIZE];

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char validate_image_char[NUM_VALIDATE][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];

unsigned char train_label_char[NUM_TRAIN][1];
unsigned char validate_label_char[NUM_VALIDATE][1];
unsigned char test_label_char[NUM_TEST][1];

static void switch_magic_endiannes(unsigned char*);

static void image_char2float(int, unsigned char [][SIZE],float [][SIZE]);
static void label_char2int(int, unsigned char [][1],int []);

/* Switch Endianness of 4 byte Magic Number */
static void switch_magic_endiannes(unsigned char *ptr)
{
	register unsigned char val;

	// Swap 1st and 4th bytes
	val = *(ptr);
	*(ptr) = *(ptr + 3);
	*(ptr + 3) = val;

	// Swap 2nd and 3rd bytes
	ptr += 1;
	val = *(ptr);
	*(ptr) = *(ptr + 1);
	*(ptr + 1) = val;
}

/* Read MNIST Images from the dataset directory,
 * Load images and labels for testing and training
 */
void read_mnist(char* file_path, int num_data, int offset, int len_info,
					int arr_n, unsigned char data_char[][arr_n], int info_arr[])
{
	FILE* fd;
	unsigned char* ptr;

	if ((fd = fopen(file_path, "r")) == NULL)
	{
		fprintf(stderr, "couldn't open image file\n");
		exit(-1);
	}

	fread(info_arr, sizeof(int), len_info, fd);
	fseek(fd, offset * sizeof(unsigned char), SEEK_CUR);

	// Read-in Magic Number
	// See IDX format in _reference_
	for (int i = 0; i < len_info; i++)
	{
		ptr = (unsigned char*)(info_arr + i);
		switch_magic_endiannes(ptr);
		ptr = ptr + sizeof(int);
	}

	// Read-in MNIST numbers (pixels|labels)
	for (int i = 0; i < num_data; i++)
	{
		fread(data_char[i], sizeof(unsigned char), arr_n, fd);
	}

	fclose(fd);
}

/* Convert Image Data from Character to float
 * by normalising it to between 0 and 1
 */
static void image_char2float(int num_data,
					unsigned char data_image_char[][SIZE],
					float data_image[][SIZE])
{
	for (int i = 0; i < num_data; i++)
		for (int j = 0; j < SIZE; j++)
			data_image[i][j] = (float)data_image_char[i][j] / 256.0;
}

/* Convert Label from Character to Integer */
static void label_char2int(int num_data,
			unsigned char data_label_char[][1],
			int data_label[])
{
	for (int i = 0; i < num_data; i++)
		data_label[i] = (int)data_label_char[i][0];
}

/* Load an Image data into image */
void load_image(float images[][SIZE], int index)
{
	for (int i = 0; i < SIZE; i++)
		image[i] = images[index][i];
}

/* Load an Image data from Training into image */
void load_train_image(int index)
{
	for (int i = 0; i < SIZE; i++)
		image[i] = train_images[train_indexer[index]][i];
}

/* Get Train image label */
int get_train_label(int index)
{
	return train_labels[train_indexer[index]];
}

/* Shuffle train indixes */
void shuffle_train_indexes()
{
	// Fisher-Yates shuffle
	// https://w.wiki/5ttL
	for (int i = 0; i <= NUM_TRAIN - 2; i++)
	{
		int j = i + (rand() % (NUM_TRAIN - i));
		int temp = train_indexer[i];
		train_indexer[i] = train_indexer[j];
		train_indexer[j] = temp;
	}
}

/* Load the data from the 4 MNIST IDX files */
void load_mnist()
{
	read_mnist(TRAIN_IMAGE, NUM_TRAIN, 0, LEN_INFO_IMAGE, SIZE,
		train_image_char, info_image);
	image_char2float(NUM_TRAIN,
		train_image_char, train_images);

	read_mnist(TRAIN_IMAGE, NUM_VALIDATE, NUM_TRAIN, LEN_INFO_IMAGE, SIZE,
		validate_image_char, info_image);
	image_char2float(NUM_VALIDATE,
		validate_image_char, validate_images);

	read_mnist(TEST_IMAGE, NUM_TEST, 0, LEN_INFO_IMAGE, SIZE,
		test_image_char, info_image);
	image_char2float(NUM_TEST, test_image_char, test_images);

	read_mnist(TRAIN_LABEL, NUM_TRAIN, 0, LEN_INFO_LABEL,
		1, train_label_char, info_label);
	label_char2int(NUM_TRAIN, train_label_char, train_labels);

	read_mnist(TRAIN_LABEL, NUM_VALIDATE, NUM_TRAIN, LEN_INFO_LABEL,
		1, validate_label_char, info_label);
	label_char2int(NUM_VALIDATE, validate_label_char, validate_labels);

	read_mnist(TEST_LABEL, NUM_TEST, 0, LEN_INFO_LABEL,
		1, test_label_char, info_label);
	label_char2int(NUM_TEST, test_label_char, test_labels);

	for (int i = 0; i < NUM_TRAIN; i++)
		train_indexer[i] = i;
}

#endif
