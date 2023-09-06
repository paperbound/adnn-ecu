import argparse

import json
import matplotlib.pyplot as plt

shapes = ['2', '4', '8', '16,16', '32', '32,32,32', '48,48',
          '64', '64,16', '72', '96,96', '96', '128']

c = "c-math.h"
eigen = "cpp-eigen"
torch = "cpp-libtorch"
numpy = "python-numpy"

implementations = [c, eigen, torch, numpy]

shape = "shape"
ncorrect = "corrects"
stime = "system time"
utime = "user time"

# argument parser
parser = argparse.ArgumentParser(description="comparing hdrnn implementations")
parser.add_argument('-ar', '--accuracy_results', dest='afile',
                    action='store', default='accuracy_results.json')
parser.add_argument('-er', '--exectime_results', dest='efile',
                    action='store', default='results-3.json')
a = parser.parse_args()

a_results = json.load(open(a.afile)) # see sample_results_accuracy.json
e_results = json.load(open(a.efile)) # see sample_results_exec.json

accuracy_stats = {}
execution_stats = {}

for i in implementations:
    accuracy_stats[i] = {}
    execution_stats[i] = {}
    for r in a_results[i]:
        if r[shape] in shapes:
            accuracy_stats[i][r[shape]] = r[ncorrect][-1]
    for r in e_results[i]:
        if r[shape] in shapes:
            execution_stats[i][r[shape]] = r[stime] + r[utime]

def compare(prog_a, prog_b, shapes, stats):
    comparison = []
    for s in shapes:
        comparison.append(100 * (1 - (stats[prog_a][s]/stats[prog_b][s])))
    return comparison

# Comparing Accuracies

cshapes = ['2', '4', '32', '64', '96', '128']

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(torch, c, cshapes, accuracy_stats))
ax[0].set_xlabel('vs c-math.h', fontsize=9)

ax[1].barh(cshapes, compare(torch, eigen, cshapes, accuracy_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs cpp-eigen', fontsize=9)

ax[2].barh(cshapes, compare(torch, numpy, cshapes, accuracy_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs cpp-libtorch', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Accuracies of cpp-libtorch")

plt.savefig('comparison-accuracy-libtorch.png')
plt.savefig('comparison-accuracy-libtorch.pgf', bbox_inches='tight', backend='pgf')

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(c, torch, cshapes, accuracy_stats))
ax[0].set_xlabel('vs cpp-libtorch', fontsize=9)

ax[1].barh(cshapes, compare(c, eigen, cshapes, accuracy_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs cpp-eigen', fontsize=9)

ax[2].barh(cshapes, compare(c, numpy, cshapes, accuracy_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs python-numpy', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Accuracies of c-math.h")

plt.savefig('comparison-accuracy-cmath.png')
plt.savefig('comparison-accuracy-cmath.pgf', bbox_inches='tight', backend='pgf')

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(eigen, torch, cshapes, accuracy_stats))
ax[0].set_xlabel('vs cpp-libtorch', fontsize=9)

ax[1].barh(cshapes, compare(eigen, c, cshapes, accuracy_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs c-math.h', fontsize=9)

ax[2].barh(cshapes, compare(eigen, numpy, cshapes, accuracy_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs python-numpy', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Accuracies of cpp-eigen")

plt.savefig('comparison-accuracy-eigen.png')
plt.savefig('comparison-accuracy-eigen.pgf', bbox_inches='tight', backend='pgf')

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(numpy, c, cshapes, accuracy_stats))
ax[0].set_xlabel('vs c-math.h', fontsize=9)

ax[1].barh(cshapes, compare(numpy, eigen, cshapes, accuracy_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs cpp-eigen', fontsize=9)

ax[2].barh(cshapes, compare(numpy, torch, cshapes, accuracy_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs cpp-libtorch', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Accuracies of python-numpy")

plt.savefig('comparison-accuracy-numpy.png')
plt.savefig('comparison-accuracy-numpy.pgf', bbox_inches='tight', backend='pgf')

# Comparing Execution times

cshapes = ['2', '4', '32', '72', '96,96', '128']

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(torch, c, cshapes, execution_stats))
ax[0].set_xlabel('vs c-math.h', fontsize=9)

ax[1].barh(cshapes, compare(torch, eigen, cshapes, execution_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs cpp-eigen', fontsize=9)

ax[2].barh(cshapes, compare(torch, numpy, cshapes, execution_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs cpp-libtorch', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Execution Times of cpp-libtorch")

plt.savefig('comparison-exectime-libtorch.png')
plt.savefig('comparison-exectime-libtorch.pgf', bbox_inches='tight', backend='pgf')

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(c, torch, cshapes, execution_stats))
ax[0].set_xlabel('vs cpp-libtorch', fontsize=9)

ax[1].barh(cshapes, compare(c, eigen, cshapes, execution_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs cpp-eigen', fontsize=9)

ax[2].barh(cshapes, compare(c, numpy, cshapes, execution_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs python-numpy', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Execution Times of c-math.h")

plt.savefig('comparison-exectime-cmath.png')
plt.savefig('comparison-exectime-cmath.pgf', bbox_inches='tight', backend='pgf')

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(eigen, torch, cshapes, execution_stats))
ax[0].set_xlabel('vs cpp-torch', fontsize=9)

ax[1].barh(cshapes, compare(eigen, c, cshapes, execution_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs c-math.h', fontsize=9)

ax[2].barh(cshapes, compare(eigen, numpy, cshapes, execution_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs python-numpy', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Execution Times of cpp-eigen")

plt.savefig('comparison-exectime-eigen.png')
plt.savefig('comparison-exectime-eigen.pgf', bbox_inches='tight', backend='pgf')

fig, ax = plt.subplots(1, 3)

ax[0].barh(cshapes, compare(numpy, c, cshapes, execution_stats))
ax[0].set_xlabel('vs c-math.h', fontsize=9)

ax[1].barh(cshapes, compare(numpy, eigen, cshapes, execution_stats), color='xkcd:melon')
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlabel('vs cpp-eigen', fontsize=9)

ax[2].barh(cshapes, compare(numpy, torch, cshapes, execution_stats), color='xkcd:aqua')
ax[2].get_yaxis().set_visible(False)
ax[2].set_xlabel('vs cpp-libtorch', fontsize=9)

plt.setp(ax[0].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[1].get_xticklabels(), fontsize=9, horizontalalignment='right')
plt.setp(ax[2].get_xticklabels(), fontsize=9, horizontalalignment='right')

fig.suptitle("Comparing Execution Times of python-numpy")

plt.savefig('comparison-exectime-numpy.png')
plt.savefig('comparison-exectime-numpy.pgf', bbox_inches='tight', backend='pgf')


