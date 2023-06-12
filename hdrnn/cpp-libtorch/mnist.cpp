#include <torch/torch.h>
#include <gflags/gflags.h>

DEFINE_bool(quiet, false, "Path to the training data for HDR-NN");
DEFINE_string(shape, "32", "Shape of the HDR-NN");
DEFINE_uint32(epochs, 30, "The number of epochs to train the network");
DEFINE_uint32(batch_size, 10, "The batch size while performing mSGD");
DEFINE_double(learning_rate, 3.0, "The learning rate for SGD");
DEFINE_string(net, "net.pt", "File path for .pt file");
DEFINE_string(image, "1.pgm", "PGM image file path");
DEFINE_string(mnist, "./dataset", "MNIST dataset");

const unsigned int IMAGE_SIZE = 784;
const unsigned int DIGITS = 10;

const char *help = "\n"
	"\t<command> [<args>]\n\nCommands:\n"
	"\t\033[1minfer\033[22m [--image IMAGE_PATH] [--net net.pt]\n"
	"\t\033[1mtrain\033[22m [--shape 32] [--epochs 30]\n"
	"\t\033\t\033 [--quiet] [--batch_size=10]\n"
	"\t\033\t\033 [--learning_rate 3] [--net net.pt]\n";

enum command { INFER, TRAIN, NONE };

/* Parse command */
enum command parseCommand(char *c)
{
	if (c != NULL && strcmp("train", c) == 0)
	{
		return TRAIN;
	}
	else if (c != NULL && strcmp("infer", c) == 0)
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
					std::stoi(shape_str.substr(index), &n_size))};
			shape.push_back(i);
			index = index + n_size + 1;
		}
		catch (std::invalid_argument const &ex)
		{
			std::cerr << "std::invalid_argument::what(): " << ex.what() << help;
			std::exit(EXIT_FAILURE);
		}
		catch (std::out_of_range const &ex)
		{
			std::cout << "std::out_of_range::what(): " << ex.what() << help;
			std::exit(EXIT_FAILURE);
		}
	}

	return shape;
}

// Define a new Module.
struct Net : torch::nn::Module
{
	Net(std::vector<unsigned int> shape)
	{
		// size of image in row vector form
		unsigned int previous_dim = IMAGE_SIZE;
		unsigned int index = 1;
		torch::nn::Linear fc{nullptr};

		// iterate over the dimensions required in the hidden layer
		for (auto c_dim : shape)
		{
			// Construct and register Linear submodules.
			fc = register_module(
				"fc" + std::to_string(index),
				torch::nn::Linear(previous_dim, c_dim));
			previous_dim = c_dim;
			index++;
			fc_list.push_back(fc);
		}

		// finally, add the last layer
		fc = register_module(
			"fc" + std::to_string(index),
			torch::nn::Linear(previous_dim, DIGITS));
		fc_list.push_back(fc);
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x)
	{
		// Use one of many tensor manipulation functions.
		x = x.reshape({x.size(0), 784});
		for (auto fc : fc_list)
			x = torch::sigmoid(fc->forward(x));

		return x;
	}

	// Use one of many "standard library" modules.
	std::vector<torch::nn::Linear> fc_list{};
};

/* Tests then prints HDRNN network accuracy
 *
 * net       - hdrnn neural network model to test
 */
void test_network(std::shared_ptr<Net> const&net)
{
	double test_loss = 0;
	long correct = 0;

	static auto test_data_loader = torch::data::make_data_loader(
				torch::data::datasets::MNIST(FLAGS_mnist,
					torch::data::datasets::MNIST::Mode::kTest)
						.map(torch::data::transforms::Stack<>()));

	for (const auto &batch : *test_data_loader)
	{
		torch::Tensor prediction = net->forward(batch.data);
		torch::Tensor target = torch::one_hot(batch.target, 10)
								.to(torch::kFloat32);
		torch::Tensor loss = torch::mse_loss(
			prediction,
			target);
		test_loss += loss.item<float>();
		correct += (
			target.argmax(1).item<int>() -
			prediction.argmax(1).item<int>() == 0);
	}

	std::cout << "Loss: " << test_loss
			<< "| Correct: " << correct << std::endl;
}

int main(int argc, char *argv[])
{
	gflags::SetUsageMessage(help);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	torch::set_num_threads(1);
	torch::set_num_interop_threads(1);

	// TODO: allow after porting mnist.h
	// std::ios_base::sync_with_stdio(false);

	std::vector<unsigned int> shape = parseShape(FLAGS_shape);

	switch (parseCommand(argv[1]))
	{
		case INFER:
		{
			// Create a new Net.
			auto net = std::make_shared<Net>(shape);
			torch::load(net, FLAGS_net);
			test_network(net);

			// TODO : add image inference here
			break;
		}
		case TRAIN:
		{
			// Create a new Net.
			auto net = std::make_shared<Net>(shape);

			auto data_loader = torch::data::make_data_loader(
				torch::data::datasets::MNIST(FLAGS_mnist)
				.map(torch::data::transforms::Stack<>()),
					/*batch_size=*/FLAGS_batch_size);

			// Instantiate an SGD optimization algorithm to
			// update our Net's parameters.
			torch::optim::SGD optimizer(net->parameters(),
										FLAGS_learning_rate);

			for (size_t epoch = 1; epoch <= FLAGS_epochs; ++epoch)
			{
				size_t batch_index = 0;

				// Iterate the data loader to yield batches from the dataset.
				for (auto &batch : *data_loader)
				{
					// Reset gradients.
					optimizer.zero_grad();
					// Execute the model on the input data.
					torch::Tensor prediction = net->forward(batch.data);
					torch::Tensor target = torch::one_hot(
											batch.target, 10)
											.to(torch::kFloat32);
					// Compute a loss value
					// to judge the prediction of our model.
					torch::Tensor loss = torch::mse_loss(
						prediction,
						target);
					// Compute gradients of the loss w.r.t.
					// the parameters of our model.
					loss.backward();
					// Update the parameters based on the calculated gradients.
					optimizer.step();
				}

				if (!FLAGS_quiet)
				{
					std::cout << "Epoch: " << epoch << " | ";
					test_network(net);
				}
			}

			if (!FLAGS_quiet)
				torch::save(net, "net.pt");

			break;
		}
		case NONE:
		{
			std::cout << "nothing to do" << help;
		}
	}
}
