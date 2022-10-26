# Handwritten Digit Recognizer using Neural Network

Note that the data set is not added to repo, find it online :)

Alternatively, run `get-mnist.sh`

## TODO

- [ ] Move configurations to a seperate header
- [ ] Fill up README.md

## Compiling

`make`

## Generate Sample Image

Once the MNIST dataset is downloaded, run `make sample-image`

To generate a different image run,

```
./bin/genimage <INDEX_INTO_TEST_IMAGES> <PATH_TO_OUTPUT_PGM_FILE>
```

## Training

`./bin/hdrnn -train <epochs> -weights <path_to_weights>`

## Inference

`./bin/hdrnn -infer <path_to_image> -weights <path_to_weights>`
