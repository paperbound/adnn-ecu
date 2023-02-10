/*
 * HDRNN
 * @author Prasanth Thomas Shaji
 *
 */

#include <filesystem>
#include <gflags/gflags.h>
#include <iostream>

#include "hdrnn.h"

DEFINE_string(train, "", "Path to the training data for HDRNN");
DEFINE_string(weights, "./weights", "File path to dump HDRNN weights");
DEFINE_string(bias, "./bias", "File path to dump HDRNN biases");

namespace fs = std::__fs::filesystem;

int main(int argc, char* argv[])
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
		// attempt to find the dataset directory
		fs::file_status dd_status = fs::status(FLAGS_train);
		if (!fs::exists(dd_status))
		{
			std::cerr << "Could not find the dataset" << std::endl;
			return -1;
		}
		// network.train_hdrnn();
		// network.dump_hdrnn(FLAGS_weights, FLAGS_bias);
		trained = true;
	}

	if (argc >= 2) // infer image
	{
		if (!trained)
		{
			// attempt to load weights and biases into the network
			fs::file_status w_status = fs::status(FLAGS_weights);
			fs::file_status b_status = fs::status(FLAGS_bias);
			if (!fs::exists(w_status) || !fs::exists(b_status))
			{
				std::cerr << "Could not find the files to load" << std::endl;
				return -1;
			}
			network.load_hdrnn(FLAGS_weights, FLAGS_bias);
		}

		// attempt to find the image file
		for (unsigned int i = 1; i < argc; i++)
		{
			std::string filename(argv[i]);
			fs::file_status i_status = fs::status(filename);
			if (!fs::exists(i_status))
			{
				std::cerr << "Could not find image file: " << filename << std::endl;
			} else {
				network.infer_pgm_image_from_path(filename);
			}
		}
	}

	return 0;
}
