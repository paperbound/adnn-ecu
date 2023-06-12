/*
 * HDRNN
 * @author Prasanth Thomas Shaji
 *
 */

#include <iostream>
#include <cstring>
#include <sys/stat.h>

#include "hdrnn.h"

DEFINE_bool   (quiet, false, "Path to the training data for HDR-NN");
DEFINE_string (shape, "16,16", "Shape of the HDR-NN");
DEFINE_uint32 (epochs, 30, "The number of epochs to train the network");
DEFINE_uint32 (batch_size, 10, "The batch size while performing mSGD");
DEFINE_double (learning_rate, 3.0, "The learning rate for SGD");
DEFINE_string (net, "cpp-eigen.nn", "File path for HDR-NN description file");
DEFINE_string (image, "1.pgm", "PGM image file path");
DEFINE_string (mnist, "./dataset/", "MNIST dataset");

const char *help = "\n"
	"\t<command> [<args>]\n\nCommands:\n"
	"\t\033[1minfer\033[22m [--image IMAGE_PATH] [--net c-math.nn]\n"
	"\t\033[1mtrain\033[22m [--shape 16,16] [--epochs 4]\n"
	"\t\033\t\033 [--quiet] [--batch_size=10]\n"
	"\t\033\t\033 [--learning_rate 3] [--net c-math.nn]\n";

enum command {INFER, TRAIN, NONE};

/* Parse command */
enum command parseCommand(char *c)
{
	if (c != NULL && strcmp("train", c) == 0)
	{
		return TRAIN;
	} else if (c != NULL && strcmp("infer", c) == 0)
	{
		return INFER;
	}

	return NONE;
}

std::vector<unsigned int> parseShape(std::string shape_str)
{
	std::size_t n_size{}, index{};
	std::vector<unsigned int> shape;

	while (index < shape_str.length())
	{
		try
		{
			const unsigned int i{
				static_cast<unsigned int>(
					std::stoi(shape_str.substr(index), &n_size)
				)};
			shape.push_back(i);
			index = index + n_size + 1;
		}
		catch (std::invalid_argument const& ex)
		{
			std::cerr << "std::invalid_argument::what(): " << ex.what() << help;
			std::exit(EXIT_FAILURE);
		}
		catch (std::out_of_range const& ex)
		{
			std::cout << "std::out_of_range::what(): " << ex.what() << help;
			std::exit(EXIT_FAILURE);
		}
	}

	return shape;
}

int main(int argc, char *argv[])
{
	gflags::SetUsageMessage(help);

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// TODO: allow after porting mnist.h
	// std::ios_base::sync_with_stdio(false);

	std::vector<unsigned int> shape = parseShape(FLAGS_shape);

	switch(parseCommand(argv[1]))
	{
		case INFER:
		{
			hdrnn network{};
			network.load_hdrnn(FLAGS_net);
			mnist_loader::load_mnist(FLAGS_mnist);
			network.evaluate_hdrnn();
			network.infer_pgm_image_from_path(FLAGS_image);
			break;
		}
		case TRAIN:
		{
			hdrnn network(
				shape,
				FLAGS_epochs,
				FLAGS_batch_size,
				FLAGS_learning_rate
			);

			network.train_hdrnn(FLAGS_mnist);
			if (!FLAGS_quiet)
				network.dump_hdrnn(FLAGS_net);
			break;
		}
		case NONE:
		{
			std::cout << "nothing to do" << help;
		}
	}

	return 0;
}
