/*
 * Handwritten Digit Recognizer using Neural Network
 * based on Le Cunn Et. al
 *
 * @author Prasanth Thomas Shaji
 */

#include "arguments.h"

extern char *progname;

extern char *ifile;

extern bool quiet;
extern char *nfile;

extern unsigned int depth;
extern unsigned int *shape;

extern unsigned int epochs;
extern unsigned int batchSize;
extern float eta;

/* Main Program: See README.md */
int main(int argc, char *argv[])
{
	Network *hdrnn = (Network*) calloc(1, sizeof(Network));

	progname = argv[0];
	parseArguments(argc, argv);
	switch(parseCommand(argv[1]))
	{
		case INFER:
			loadHDRNN(hdrnn);
			inferImage(hdrnn);
			break;
		case TRAIN:
			load_mnist();
			initHDRNN(hdrnn);
			generate_random_weights(hdrnn);
			trainHDRNN(hdrnn, quiet);
			if (!quiet)
				dumpWeights(hdrnn);
			break;
		case NONE:
			printf("nothing to do\n");
			showHelpMessageThenExit();
			break;
	}
	return 0;
}
