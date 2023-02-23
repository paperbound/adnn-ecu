# Handwritten Digit Recognition using Neural Network in CPP + Eigen

Note that the dataset is not added to the repo, find it online

Alternatively run the `get-mnist.sh` script in `hdrnn/c-math.h`

If the folder is already setup somewhere, the path to the folder maybe
specified while training the model after the `-train` flag. See more
in Usage

## Compiling

Please note that you need to create a bin folder (`mkdir bin`)
inside `hdrnn/cpp-eigen/` much like in `hdrnn/c-math.h/`

This project uses cmake

Firstly, create a build directory. E.g. `mkdir build`. Further commands
to run while building the project should be run in this build folder

Specify the source directory to cmake: `cmake <path_to_source>`, where
path_to_source is the `hdrnn/cpp-eigen/` directory containing the
CMakeLists.txt file

Now you can use autotools (cmake default) to build the project:
`cmake --build .`

The binary will be written to the `bin/` folder created in the project
source directory

## Usage Comments

For some helpful usage comments, you may run the program with a `-h` flag

The usage is similar to that described in `hdrnn/c-math.h/README.md`
