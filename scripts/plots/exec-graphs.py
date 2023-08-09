import argparse
import json
import matplotlib.pyplot as plt

IMAGE_SIZE = 784
DIGITS     = 10

PROGRAM  = "program"
EPOCHS   = "epochs"
SHAPE    = "shape"
SIZE     = "size"
USERTIME = "user time"
SYSTIME  = "system time"
MSS      = "maximum resident set size"

parser = argparse.ArgumentParser(description="graphing for hdrnn measurements")
parser.add_argument('-r', '--results', dest='rfile',
        action='store', default="results.json")
a = parser.parse_args()

# see sample_results json
results = json.load(open(a.rfile))

plt.rcParams.update(
    {
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'figure.figsize': (6.7, 4.2),
        'axes.titlesize': 'small',
        'font.size': 10,
    }
)

fig, ax = plt.subplots()

plt.xlabel("HDR-NN hidden layer shape")
plt.ylabel("Time to complete 1 Epoch of Training (seconds)")

for program in results:
    measures = [ (r[SIZE], r[USERTIME] + r[SYSTIME], r[SHAPE])
                  for r in results[program] ]
    measures = sorted(measures, key=lambda measure: measure[0])
    ax.plot([ m[0] for m in measures ],
            [ m[1] for m in measures ],
            label=program, marker='v')
    ax.set_xticks([ m[0] for m in measures ],
                  [ m[2] for m in measures ])


plt.setp(ax.get_xticklabels(), fontsize=9, rotation=42, horizontalalignment='right')
plt.legend()
plt.savefig('exec-time.pgf', bbox_inches='tight', backend='pgf')

#plt.show()
