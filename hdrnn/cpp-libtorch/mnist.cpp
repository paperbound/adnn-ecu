#include <torch/torch.h>

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 30));
    fc2 = register_module("fc2", torch::nn::Linear(30, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::sigmoid(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::sigmoid(fc2->forward(x));
    //return x.argmax(1).to(torch::kFloat32); Using softmax instead of argmax
    return x.softmax(1, torch::kFloat32);
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
  // Create a new Net.
  auto net = std::make_shared<Net>();

  // Create a multi-threaded data loader for the MNIST dataset.
  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./data").map(
          torch::data::transforms::Stack<>()),
      /*batch_size=*/10);

  auto test_data_loader = torch::data::make_data_loader(
	torch::data::datasets::MNIST("./data",
	     torch::data::datasets::MNIST::Mode::kTest).map(
		    torch::data::transforms::Stack<>()));

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/3.0);

  for (size_t epoch = 1; epoch <= 30; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = net->forward(batch.data);
      torch::Tensor target = torch::one_hot(batch.target, 10).to(torch::kFloat32);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::mse_loss(
					   prediction,
					   target
					   );
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
    }

    // Test, output loss, and checkpoint every epoch
    double test_loss = 0;
    long correct = 0;
    for (const auto& batch : *test_data_loader) {
	    torch::Tensor prediction = net->forward(batch.data);
	    torch::Tensor target = torch::one_hot(batch.target, 10)
		    .to(torch::kFloat32);
	    torch::Tensor loss = torch::mse_loss(
						 prediction,
						 target
						 );
	    test_loss += loss.item<float>();
	    correct += (target.argmax(1).item<int>() - prediction.argmax(1).item<int>() == 0);
    }

    // Output loss and checkpoint every epoch
    std::cout << "Epoch: " << epoch << " | Loss: " << test_loss
	      << "| Correct: " << correct << std::endl;
    // Serialize your model periodically as a checkpoint.
    torch::save(net, "net.pt");
  }
}
