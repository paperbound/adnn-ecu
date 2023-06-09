# General Implementation Notes

Primary Work ongoing for Eigen based CPP implementation

Benchmarking activity is currenlty pending

## General Design

- [x] Rethink the weights/biases file format
- [ ] Add Doxygen / other documentation into the hdrnn codebase
- [ ] Quantisation and Pruning in c-math.h
- [x] perf tests on EB
- [x] Heaptrack on EB

## c-math.h

C based implementation of HDRNN

```
./c-math/hdrnn # to view usage instructions
```

### Task Checklist

- [x] Complete an inference run
- [x] Complete a training run
- [x] Test on EB / BBB
- [x] Change current (argc, argv) handling

- [x] Benchmark Training
- [ ] Profile Training

- [ ] Possibly remove lib-c dependency
  - [x] string.h only used for memset

- [ ] Remove reading the magic number in `mnist.h` (LOW PRIORITY)

## cpp-eigen

CPP implementation of HDRNN using the Eigen vector library

### Task Checklist

- [x] Complete an inference run
- [x] Create debug/release configurations into cmake
- [ ] Complete a training run (pending correctness only on M1)
- [x] Test on EB / BBB

- [x] Benchmark Training

## cpp-libtorch

CPP implementation based on PyTorch

### Task Checklist

- [x] Complete an inference run
- [x] Complete a training run
- [x] Test on EB / BBB

- [x] Benchmark Training

## rust-ndarray

Rust implementation of HDRNN using the ndarray crate

### Task Checklist

- [ ] Complete an inference run
- [ ] Complete a training run
- [ ] Test on EB / BBB

- [ ] Benchmark Training

## python-numpy

Python implementation of HDRNN using numpy taken by Michael Nielsen,
hosted on neuralnetworksanddeeplearning.com

### Task Checklist

- [x] Complete an inference run
- [x] Complete a training run
- [x] Test on EB / BBB

- [x] Benchmark Training

## python-tf

Python implementation of HDRNN using Tensorflow

### Task Checklist

- [x] Complete an inference run
- [x] Complete a training run
- [ ] Test on EB / BBB

- [ ] Benchmark Training

## python-pytorch

Python implementation of HDRNN using PyTorch

### Task Checklist

- [ ] Complete an inference run
- [ ] Complete a training run
- [ ] Test on EB / BBB

- [ ] Benchmark Training
