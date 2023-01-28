/*
 * Handwritten Digit Recognizer using Neural Network
 * based on Le Cunn Et. al
 *
 * @author Prasanth Thomas Shaji
 *
 * For Usage Information see README.md
 *
 */

#include <string.h>
#include <time.h>

#include "network.h"

extern int epochs;

/* Show a Help message for the program, then Exit */
void showHelpMessageThenExit(char *progname) {
	printf("Usage: %s [-train|-infer] (epochs|path_to_image))", progname);
	printf(" -weights <path_to_weights> -bias <path_to_bias>\n");
	printf("(DOESNT WORK RIGHT NOW) Training Example: %s -train 1000", progname);
	printf(" -weights out_weights -bias out_bias\n");
	printf("Inference Example: %s -infer tests/image1.pgm", progname);
	printf(" -weights in_weights -bias in_bias\n");
	exit(-1);
}

/* Main Program: See README.md */
int main(int argc, char *argv[]) {
	if (argc != 7) {
		showHelpMessageThenExit(argv[0]);
	}

	clock_t begin = clock();
	Network *hdrnn = (Network*) calloc(1, sizeof(Network));
	initHDRNN(hdrnn);
	if (strcmp("-train", argv[1]) == 0) {
		// Train the HDRNN
		load_mnist();
		generate_random_weights(hdrnn);
		trainHDRNN(hdrnn);
		dumpWeights(hdrnn, argv[4], argv[6]);
	} else if (strcmp("-infer", argv[1]) == 0) {
		// Infer
		loadHDRNN(hdrnn, argv[4], argv[6]);
		inferImage(hdrnn, argv[2]);
	} else {
		printf("Please enter the correct commands\n\n");
		showHelpMessageThenExit(argv[0]);
	}
	clock_t end = clock();
	printf("Total Execution Time %lfs\n", (double)(end - begin) / CLOCKS_PER_SEC);
	return 0;
}
