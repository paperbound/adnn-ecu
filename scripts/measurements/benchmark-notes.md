# HDRNN Benchmarking

Primary tool for measurements will be the linux based perf utility,
and the primary platform will be the mx6sabresd Evaluation board

## Benchmark Metrics

The primary execution measurement will be total program execution time

## HDRNN Implementations

1. python-numpy
2. c-math.h
3. cpp-eigen
4. cpp-libtorch

## Setup

Copy hdrnn directory onto mx6sabresd, ensure binaries runnable and present.

```
bash run.sh
```

### cpp-libtorch

Source libraries to appropriate folders

```
cp libgomp.so.1 /lib # GNU Open MP library
cp libgfortran.so.5 /lib # GNU Fortran compiler internals
cp libopenblas.so.1 /lib # Open Blas
```

## Measurement

Setup a screen and run `bash run.sh`
