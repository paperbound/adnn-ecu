# Handwritten Digit Recognizer using Neural Network in C

Note that the data set is not added to repo, find it online :)

Alternatively, run `get-mnist.sh`

## Compiling

Create a bin folder if not already present `mkdir bin/` inside `hdrnn/c-math.h/`

Then use the ubiquitous autotools

`make`

The binary will be written to the `bin/` folder created above

## Generate Sample Images

Once the MNIST dataset is downloaded, run `make sample-image`

To generate a different image run,

```
./bin/genimage <INDEX_INTO_TEST_IMAGES> <PATH_TO_OUTPUT_PGM_FILE>
```

Alternatively see this facility in `hdrnn/python-numpy`

## Usage Comments

Running the program without any arguments should show a useful help message

### Inference

`./bin/hdrnn -infer <path_to_image>
	-weights <path_to_weights> -bias <path_to_biases>`

You can create the images using the methods specified in this README.md.
It is assumed that the image will be a kind of PGM (2) format

The paths to the weights and biases files must contain HDRNN weights created
by the `python-numpy`, `c-math.h`, or `cpp-eigen` programs

### Training

`./bin/hdrnn -train <epochs> -weights <path_to_weights> -bias <path_to_biases>`

Currenlty, the `<epochs>` argument can be anything and is not processed, it
does need to be given while running the command however

Note that this will create files specified in the paths to dump the weights of
the trained model into
