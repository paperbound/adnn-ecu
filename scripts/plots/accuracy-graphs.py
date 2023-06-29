import argparse
import json
import matplotlib.pyplot as plt

IMAGE_SIZE = 784
DIGITS     = 10

PROGRAM  = "program"
EPOCHS   = "epochs"
SHAPE    = "shape"
SIZE     = "size"
CORRECTS = "corrects"

shapes = ['2', '4', '8', '16,16', '32', '32,32,32', '64', '96', '128']
colors = {
    '2' : 'xkcd:sky',
    '4' : 'xkcd:melon',
    '8' : 'xkcd:jade',
    '16,16' : 'xkcd:strawberry',
    '32' : 'xkcd:amethyst',
    '32,32,32' : 'xkcd:cocoa',
    '64' : 'xkcd:rosy pink',
    '96' : 'xkcd:dirty purple',
    '128' : 'xkcd:stone'
}

parser = argparse.ArgumentParser(description="graphing for hdrnn measurements")
parser.add_argument('-r', '--results', dest='rfile',
        action='store', default="results.json")
a = parser.parse_args()

# see sample_results json
results = json.load(open(a.rfile))

width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

shape_accuracies = {}
total_programs = 0

for program in results:
    total_programs += 1
    measures = [ (r[SIZE], r[CORRECTS], r[SHAPE])
                  for r in results[program] ]
    measures = sorted(measures, key=lambda measure: measure[0])
    for m in measures:
        accuracy = round((sum(m[1]) / len(m[1])) / 100, 1)
        if m[2] not in shape_accuracies.keys():
            shape_accuracies[m[2]] = []
        shape_accuracies[m[2]].append(accuracy)

xs = list(range(total_programs))

for s in shape_accuracies:
    offset = width * multiplier
    rects = ax.bar([x + offset for x in xs],
                   shape_accuracies[s], width,
                   label=s,
                   color=colors[s])
    ax.bar_label(rects, padding=4)
    multiplier += 1

ax.set_xticks([x + 4*width for x in xs], list(results.keys()))
ax.set_ylabel('Model Accuracy in %')
ax.legend(loc='upper center', ncols=len(shapes),
          bbox_to_anchor=(0.5, 1.09), title='HDR-NN Hidden Layer Shapes')
ax.set_ylim(0, 100)

plt.show()
