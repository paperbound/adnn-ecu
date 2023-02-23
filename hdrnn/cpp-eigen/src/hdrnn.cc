/*
 * HDRNN
 * @author Prasanth Thomas Shaji
 *
 */

#include <gflags/gflags.h>
#include <iostream>
#include <sys/stat.h>

#include "hdrnn.h"

DEFINE_string(train, "", "Path to the training data for HDRNN");
DEFINE_string(weights, "./weights", "File path to dump HDRNN weights");
DEFINE_string(bias, "./bias", "File path to dump HDRNN biases");

bool check_file(std::string path)
{
	struct stat buffer;
	if (stat(path.c_str(), &buffer) != 0)
	{
		std::cerr << "Could not find the file: " << path << std::endl;
		return false;
	}
	return true;
}

int main(int argc, char *argv[])
{
	gflags::SetUsageMessage("Usage: hdr [-train] [-weights (file)] "
							"[-bias (file)]");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// TODO: allow after porting mnist.h
	// std::ios_base::sync_with_stdio(false);

	hdrnn network({30});

	bool trained = false;

	if (FLAGS_train.length() > 0) // train HDRNN
	{
		// attempt to find the dataset
		if (!(
				check_file(FLAGS_train)
				&& check_file(FLAGS_train + TRAIN_IMAGE)
				&& check_file(FLAGS_train + TRAIN_LABEL)
				&& check_file(FLAGS_train + TEST_IMAGE)
				&& check_file(FLAGS_train + TEST_LABEL))
			)
			return -1;
		network.train_hdrnn(FLAGS_train);
		network.dump_hdrnn(FLAGS_weights, FLAGS_bias);
		trained = true;
	}

	if (argc >= 2) // infer image
	{
		if (!trained)
		{
			// attempt to load weights and biases into the network
			if (!check_file(FLAGS_weights) || !check_file(FLAGS_bias))
				return -1;

			network.load_hdrnn(FLAGS_weights, FLAGS_bias);
		}

		// attempt to find the image files
		for (int i = 1; i < argc; i++)
		{
			std::string filename(argv[i]);
			if (check_file(filename))
				network.infer_pgm_image_from_path(filename);
		}
	}

	return 0;
}
