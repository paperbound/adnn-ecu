import argparse
import json

PROGRAMS = ("c-math.h", "cpp-eigen", "cpp-libtorch", "python-numpy")

# see sample_log_entry
LOG_ENTRY_SIZE = 23

# mnist values
IMAGE_SIZE = 784
DIGITS     = 10

# see sample_results json
results = {}
for p in PROGRAMS:
    results[p] = []

parser = argparse.ArgumentParser(description="parsing hdrnn measurements")
parser.add_argument('-l', '--logfile', dest='lfile',
                    action='store', default="log")
parser.add_argument('-r', '--results', dest='rfile',
                    action='store', default="results.json")
a = parser.parse_args()

with open(a.lfile) as f:
    log = f.read()

log = log.split('\n')

def getSize(shape):
    shape = [ int(i) for i in shape.split(",") ]
    size = 0
    previous_dimension = IMAGE_SIZE
    for current_dimension in shape:
        size += (
            (previous_dimension * current_dimension) # weights
            + current_dimension                      # biases
            )
        previous_dimension = current_dimension
    return size

def parseCommand(command):
    '''
    Expects a string like "./c-math.h/hdrnn train --epochs 1 --quiet --shape 2"
    Returns program name, number of epochs, and shape string
        > ("c-math.h", 1, "2")
    '''
    oparse = argparse.ArgumentParser()
    oparse.add_argument('-e', '--epochs', dest='epochs',
                    action='store', default=30, type=int)
    oparse.add_argument('-s', '--shape', dest='shape',
                    action='store', default='30')
    o = oparse.parse_known_args(command.split())
    prog = ""
    found = False
    for p in PROGRAMS:
        if p in command:
            prog = p
            found = True
            break
    if not found:
        raise("Unknown program found : " + command)
    return prog, o[0].epochs, o[0].shape

for entry in [ log[i : i + LOG_ENTRY_SIZE]
              for i in range(0, len(log), LOG_ENTRY_SIZE)]:
    if (len(entry) == LOG_ENTRY_SIZE):
        # Checkout sample_log_entry to understand the numbers here
        p, e, s = parseCommand(entry[0][21:-1] + ' "')
        user_time = float(entry[1][21:])
        system_time = float(entry[2][23:])
        mrss = int(entry[9][36:])
        results[p].append({
            "epochs"                    : e,
            "shape"                     : s,
            "size"                      : getSize(s),
            "user time"                 : user_time,
            "system time"               : system_time,
            "maximum resident set size" : mrss
        })

print(json.dumps(results, sort_keys=True, indent=4))
