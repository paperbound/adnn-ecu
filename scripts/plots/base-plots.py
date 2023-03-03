from matplotlib import pyplot as plt

figsize = (22, 7)
fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=80)
plt.xscale('log')
def calc_nparams(shape):
    inputs = shape[0]
    # np = inputs
    np = 0
    for l in shape[1:]:
        np += (l + inputs * l)
        inputs = l
    return np

# c_num_params = [ calc_nparams(i) for i in [
#     [784, 10],
#     [784, 30, 10],
#     [784, 10, 10],
#     [784, 20, 10],
#     [784, 5, 5, 10],
#     [784, 10, 5, 10],
#     [784, 40, 10]
# ] ]
# c_accuracy = [
#     13.61,
#     94.65, 
#     92.14, 
#     93.53, 
#     84.68,
#     90.07, 
#     95.12
# ]
# c_exec_times = [
#     5 * 60 + 33.47,
#     22 * 60 + 57.95,
#     7 * 60 + 29.65,
#     15 * 60 + 16.04,
#     3 * 60 + 36.9,
#     7 * 60 + 38.35,
#     30 * 60 + 34.30,
# ]
c_num_params = [1600, 3190, 6370, 12730, 25450, 50890, 101770, 203530, 25818, 51450, 102714]
c_accuracy = [30.5, 56.82, 90.57, 93.42, 94.85, 87.14, 76.27, 67.98, 94.67, 95.4, 96.02]
c_exec_times = [85.08, 164.87, 351.78, 725.95, 1470.73, 2983.59, 6885.65, 15581.91, 1509.27, 3068.94, 6650.18]

numpy_num_params = [ calc_nparams(i) for i in [
    [784, 2, 10],
    [784, 4, 10],
    [784, 8, 10],
    [784, 32, 10],
    # [784, 40, 10],
] ]
numpy_accuracy = [
    35.47,
    56.72,
    90.84,
    95.37,
    # 95.18,
]
numpy_exec_times = [
    1978.8,
    2171.69,
    2589.42,
    5799.51
]


def nsort(nparams, metric):
    return zip(*sorted(zip(nparams, metric)))

xs, ys = nsort(c_num_params, c_accuracy)
ax[0].plot(xs, ys, label='C math.h')
xs, ys = nsort(numpy_num_params, numpy_accuracy)
ax[0].plot(xs, ys, label='Python Numpy')
ax[0].set_xlabel('Number of Parameters')
ax[0].set_ylabel('Accuracy (%)')
ax[0].legend()

xs, ys = nsort(c_num_params, c_exec_times)
ax[1].plot(xs, ys, label='C math.h')
xs, ys = nsort(numpy_num_params, numpy_exec_times)
ax[1].plot(xs, ys, label='Python Numpy')
ax[1].set_xlabel('Number of Parameters')
ax[1].set_ylabel('Execution Times (in seconds)')
ax[1].legend()

# print(numpy_num_params, c_num_params)

plt.show()

c_exec_times_nw = [i/60 for i in c_exec_times]
numpy_exec_times_nw = [i/60 for i in numpy_exec_times]
fig, bx = plt.subplots()
plt.xscale('linear')
bx.scatter(c_exec_times_nw, c_accuracy, label='C math.h')
bx.scatter(numpy_exec_times_nw, numpy_accuracy, label='Python Numpy')
for i, txt in enumerate(c_num_params):
    bx.annotate(txt, (c_exec_times_nw[i],c_accuracy[i]))
for i, txt in enumerate(numpy_num_params):
    bx.annotate(txt, (numpy_exec_times_nw[i],numpy_accuracy[i]))
bx.set_xlabel('Execution Times (in minutes)')
bx.set_ylabel('Accuracy (%)')
bx.legend()
plt.show()