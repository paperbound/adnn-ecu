from matplotlib import pyplot as plt

figsize = (22, 7)
fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=80)

def calc_nparams(shape):
    inputs = shape[0]
    # np = inputs
    np = 0
    for l in shape[1:]:
        np += (l + inputs * l)
        inputs = l
    return np

c_num_params = [ calc_nparams(i) for i in [
    [784, 10],
    [784, 30, 10],
    [784, 10, 10],
    [784, 20, 10],
    [784, 5, 5, 10],
    [784, 10, 5, 10],
    [784, 40, 10]
] ]
c_accuracy = [
    13.61,
    94.65, 
    92.14, 
    93.53, 
    84.68,
    90.07, 
    95.12
]
c_exec_times = [
    5 * 60 + 33.47,
    22 * 60 + 57.95,
    7 * 60 + 29.65,
    15 * 60 + 16.04,
    3 * 60 + 36.9,
    7 * 60 + 38.35,
    30 * 60 + 34.30,
]

numpy_num_params = [ calc_nparams(i) for i in [
    [784, 10],
    [784, 30, 10],
    [784, 10, 10],
    [784, 20, 10],
    [784, 40, 10],
] ]
numpy_accuracy = [
    80.76,
    95.34,
    92.04,
    94.44,
    95.18,
]
numpy_exec_times = [
    33 * 60 + 17.77,
    1 * 60 * 60 + 26 * 60 + 38,
    46 * 60 + 26.99,
    1 * 60 * 60 + 5 * 60 + 49,
    1 * 60 * 60 + 54 * 60 + 8,
]


def nsort(nparams, metric):
    return zip(*sorted(zip(nparams, metric)))

xs, ys = nsort(c_num_params, c_accuracy)
ax[0].plot(xs, ys, label='C math.h')
xs, ys = nsort(numpy_num_params, numpy_accuracy)
ax[0].plot(xs, ys, label='Python Numpy')
ax[0].set_xlabel('Number of Parameters')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

xs, ys = nsort(c_num_params, c_exec_times)
ax[1].plot(xs, ys, label='C math.h')
xs, ys = nsort(numpy_num_params, numpy_exec_times)
ax[1].plot(xs, ys, label='Python Numpy')
ax[1].set_xlabel('Number of Parameters')
ax[1].set_ylabel('Execution Times')
ax[1].legend()

plt.show()
