/*
 * @author Prasanth Thomas Shaji
 *
 * Experiments with project-hdrnn
 *
 */

#include "mnist.h"
#include "images.h"

int main(int argc, char *argv[]) {
	int index;
	if (argc != 3)
	{
		printf("Please use the format ./<prog> <test image index> <output filename>");
		exit(-1);
	} else
	{
		index = atoi(argv[1]);
	}

	load_mnist();
	write_image(argv[2], index);
	load_infer_image(argv[2]);
	return 0;
}
