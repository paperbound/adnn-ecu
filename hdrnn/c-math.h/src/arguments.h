/*
 * @author Prasanth Thomas Shaji
 *
 * HDR-NN argument parser
 *
 */

/*
 * Note that string comparison and manipulation in this programme
 * are not based on string.h and consider strings to be single byte
 * null terminated. These and other assumptions are flaky and can
 * cause other issue such as memory safety properties however
 * those concerns for the test programme are effectively ignored
 *
 * Assumed locale is "en_GB.UTF-8" (TODO: Confirm)
 */

#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include "network.h"

char *progname;

extern char *ifile;

extern bool quiet;
extern char *nfile;

extern unsigned int depth;
extern unsigned int *shape;

extern unsigned int epochs;
extern unsigned int batchSize;
extern float eta;

enum command {INFER, TRAIN, NONE} run_command = NONE;

/* commands */
const char *commands[] = {
	"infer", "train"
};

const unsigned char QUIET         = 1;
const unsigned char SHAPE         = 2;
const unsigned char EPOCHS        = 3;
const unsigned char BATCH_SIZE    = 4;
const unsigned char LEARNING_RATE = 5;
const unsigned char NET           = 6;
const unsigned char IMAGE         = 7;

/* cli arguments */
const char *arguments[] = {
	"quiet", "q",            // 5  1   (1)
	"shape", "s",            // 5  1   (2)
	"epochs", "e",           // 6  1   (3)
	"batch_size", "bs",      // 10 2   (4)
	"learning_rate", "lr",   // 13 2   (5)
	"net", "n",              // 3  1   (6)
	"image", "i"             // 5  1   (7)
};

// TODO: only allow --image / -i and --net / -n for infer command
// TODO: allow everything except --image / -i for train command

/* Return the index of the smaller argument string */
static const size_t ais(unsigned int i)
{
	return 2 * i - 1;
}

/* Return the index of the bigger argument string */
static const size_t aib(unsigned int i)
{
	return 2 * i - 2;
}

const int nargs = 7;

const char *help = "\n"
	"\t<command> [<args>]\n\nCommands:\n"
	"\t\033[1minfer\033[22m [-i, --image IMAGE_PATH] [-n, --net c-math.nn]\n"
	"\t\033[1mtrain\033[22m [-s, --shape 16,16] [-e, --epochs 4]\n"
	"\t\033\t\033 [-q, --quiet] [-bs, --batch_size=10]\n"
	"\t\033\t\033 [-lr, --learning_rate 3] [-n, --net c-math.nn]\n";

static void         parseShape(char *);
static unsigned int parseInteger(char *, size_t);
static float        parseFloat(char *);

static char * new_string_copy(char *);
static size_t get_string_length(const char *);
static bool   are_strings_matching(char *, const char *, unsigned int);

#define ENSURE_NEXT_ARG(next, count) if (next >= count) showHelpMessageThenExit();

/* Show a Help message for the program, then Exit */
void showHelpMessageThenExit() {
	printf("Usage: \033[1m%s\033[22m [--quiet]\n", progname);
	puts(help);
	exit(-1);
}

/* Understand program innvocation */
void parseArguments(int argc, char *argv[])
{
	size_t asize;

	// Ignore the first two fields
	for (int i = 2; i < argc; i++)
	{
		// Expect an argument with a dash
		if (argv[i][0] != '-')
			showHelpMessageThenExit();

		asize = get_string_length(argv[i]);

		switch (asize)
		{
			case 2: // "-q" "-s" "-e" "-n" "-i"
			{
				switch (argv[i][1])
				{
					case 'q': // arguments[ais(QUIET)]
					{
						quiet = true;
						break;
					}
					case 's': // arguments[ais(SHAPE)]
					{
						ENSURE_NEXT_ARG(++i, argc)

						parseShape(argv[i]);
						break;
					}
					case 'e': // arguments[ais(EPOCHS)]
					{
						ENSURE_NEXT_ARG(++i, argc)

						epochs = parseInteger(argv[i],
									get_string_length(argv[i]));
						break;
					}
					case 'n': // arguments[ais(NET)]
					{
						ENSURE_NEXT_ARG(++i, argc)

						nfile = new_string_copy(argv[i]);
						break;
					}
					case 'i': // arguments[ais(IMAGE)]
					{
						ENSURE_NEXT_ARG(++i, argc)

						ifile = new_string_copy(argv[i]);
						break;
					}
				}
				break;
			}
			case 3: // "-bs" "-lr"
			{
				ENSURE_NEXT_ARG(++i, argc)

				if (are_strings_matching(argv[i-1] + 1,
						arguments[ais(BATCH_SIZE)],
						asize))
				{
					batchSize = parseInteger(argv[i],
									get_string_length(argv[i]));
				} else if (are_strings_matching(argv[i-1] + 1,
								arguments[ais(LEARNING_RATE)],
								asize))
				{
					eta = parseFloat(argv[i]);
				} else
				{
					showHelpMessageThenExit();
				}
				break;
			}
			case 5: // "--net"
			{
				ENSURE_NEXT_ARG(++i, argc)

				if (are_strings_matching(argv[i-1] + 2,
						arguments[aib(NET)],
						asize))
				{
					nfile = new_string_copy(argv[i]);
				} else {
					showHelpMessageThenExit();
				}
				break;
			}
			case 7: // "--quiet" "--shape" "--image"
			{
				if (are_strings_matching(argv[i-1] + 2,
						arguments[aib(QUIET)],
						asize))
				{
					quiet = true;
				} else if (are_strings_matching(argv[i-1] + 2,
								arguments[aib(SHAPE)],
								asize))
				{
					ENSURE_NEXT_ARG(++i, argc)

					parseShape(argv[i]);
				} else if (are_strings_matching(argv[i-1] + 2,
								arguments[aib(IMAGE)],
								asize))
				{
					ENSURE_NEXT_ARG(++i, argc)

					ifile = new_string_copy(argv[i]);
				} else {
					showHelpMessageThenExit();
				}
				break;
			}
			case 8: // "--epochs"
			{
				ENSURE_NEXT_ARG(++i, argc)

				if (are_strings_matching(argv[i-1] + 2,
						arguments[aib(EPOCHS)],
						asize))
				{
					epochs = parseInteger(argv[i],
									get_string_length(argv[i]));
				} else {
					showHelpMessageThenExit();
				}
				break;
			}
			case 12: // "--batch_size"
			{
				ENSURE_NEXT_ARG(++i, argc)

				if (are_strings_matching(argv[i-1] + 2,
						arguments[aib(BATCH_SIZE)],
						asize))
				{
					batchSize = parseInteger(argv[i] ,
									get_string_length(argv[i]));
				} else {
					showHelpMessageThenExit();
				}
				break;
			}
			case 15: // "--learning_rate"
			{
				ENSURE_NEXT_ARG(++i, argc)

				if (are_strings_matching(argv[i-1] + 2,
						arguments[aib(LEARNING_RATE)],
						asize))
				{
					eta = parseFloat(argv[i]);
				} else {
					showHelpMessageThenExit();
				}
				break;
			}

		}
	}
}

/* Parse a real number argument*/
static float parseFloat(char *f_str)
{
	float value = 0.0;
	unsigned int exp = 0;

	unsigned int i, size, digit;
	size = get_string_length(f_str);

	for (i = 0; i < size; i++)
	{
		digit = f_str[i] - '0';
		if (digit >= 0 && digit <= 9)
		{
			value = value * 10.0 + digit;
		}
		else
		{
			break;
		}
	}

	if (f_str[i++] == '.')
	{
		for (; i < size; i++)
		{
			digit = f_str[i] - '0';
			if (digit >= 0 && digit <= 9)
			{
				value = value * 10.0 + digit;
				exp++;
			}
			else
			{
				showHelpMessageThenExit();
			}
		}
	}

	for (i = 0; i < exp; i++)
	{
		value = value * 0.1;
	}

	return value;
}

/* Parse a natural number argument */
static unsigned int parseInteger(char *n_str, size_t size)
{
	unsigned int value, digit;
	value = 0;

	for (size_t j = 0; j < size; j++)
	{
		digit = n_str[j] - '0';
		if (digit >= 0 && digit <= 9)
			value = value*10 + digit;
		else
			showHelpMessageThenExit();
	}
	return value;
}

/* Parse network shape
 *
 * Set the value in network.h's shape variable */
static void parseShape(char *shape_str)
{
	// Count the ','s
	unsigned int str_size, n_size = 1;
	str_size = get_string_length(shape_str);
	for (unsigned int i = 0; i < str_size; i++)
	{
		if (shape_str[i] == ',')
			n_size += 1;
	}

	// Get the shape
	shape = (unsigned int *) calloc(n_size + 2, sizeof(int));
	unsigned int value, index, last;

	last = 0;
	shape[0] = SIZE; // input layer size
	index = 1;

	for (unsigned int i = 0; i < str_size; i++)
	{
		if (shape_str[i] == ',')
		{
			value = parseInteger(shape_str+last, i-last);
			shape[index] = value;
			index += 1;
			last = i + 1;
		}
	}
	value = parseInteger(shape_str+last, str_size-last);
	shape[index] = value;
	index += 1;

	if (index != n_size + 1)
		showHelpMessageThenExit();

	shape[index] = DIGITS; // output layer size

	depth = n_size + 2; // Setting network depth
}

/* Parse command */
enum command parseCommand(char *c)
{
	/* There are only 2 possible commands,
	 * each with length 5 */
	if (c != NULL && get_string_length(c) == 5)
	{
		if (are_strings_matching(c, commands[0], 5))
		{
			return INFER;
		} else if (are_strings_matching(c, commands[1], 5))
		{
			return TRAIN;
		}
	}

	return NONE;
}

/* Allocate a new string on heap */
static char * new_string_copy(char *original)
{
	size_t size = get_string_length(original);
	char * new_string = (char *) calloc(size, sizeof(char));
	for (size_t i = 0; i < size; i++)
		new_string[i] = original[i];
	return new_string;
}

/* Get string length */
static size_t get_string_length(const char *str)
{
	const char *a = str;
	for (; *str; str++);
	return (str - a);
}

/* Check if the strings are matching */
static bool are_strings_matching(char *a, const char *b, unsigned int size)
{
	for (unsigned int i = 0; i < size; i++)
		if (a[i] ^ b[i])
			return false;
	return true;
}

#endif
