# General Implementation Notes

Primary Work ongoing for Eigen based CPP implementation

Benchmarking activity is currenlty pending

## General Design

- [ ] Rethink the weights/biases file format
- [ ] Add Doxygen / other documentation into the hdrnn codebase
- [ ] Have a plan for implementing quantisation and pruning

## python-numpy

### Task Checklist
- [x] Complete an inference run
- [x] Complete a training run
- [x] Test on EB / BBB
- [ ] Change current (argc, argv) handling
- [ ] Figure out perf tests
- [ ] Benchmark Training

## python-tf

### Task Checklist
- [x] Complete an inference run
- [x] Complete a training run
- [ ] Test on EB / BBB
- [ ] Change current (argc, argv) handling
- [ ] Figure out perf tests
- [ ] Benchmark Training

## c-math.h

C based implementation of HDRNN

```
./c-math/hdrnn # to view usage instructions
```

### Task Checklist

- [x] Complete an inference run
- [x] Complete a training run
- [x] Test on EB / BBB
- [ ] Change current (argc, argv) handling
- [ ] Figure out perf tests
- [ ] Benchmark Training

## cpp-eigen

CPP based implementation of HDRNN using the Eigen vector library

### Task Checklist

- [ ] Complete an inference run
- [ ] Create debug/release configurations into cmake
- [ ] Complete a training run
- [ ] Test on EB / BBB
- [ ] Change current (argc, argv) handling
- [ ] Figure out perf tests
- [ ] Benchmark Training
