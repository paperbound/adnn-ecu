"""
parse.py
@author Prasanth Shaji

Takes a log output from one of the measurement scripts and outputs a JSON
JSON can then be used by one of the graph scripts
"""

import argparse
import json
import re

PROGRAMS = ("c-math.h", "cpp-eigen", "cpp-libtorch", "python-numpy")

# log entry types
LOG = "exec"
ALOG = "accuracy"
LOG_TYPES = (LOG, ALOG)

# see sample_log_entry
LOG_ENTRY_SIZE = 23
ALOG_ENTRY_SIZE = 15

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
parser.add_argument('-t', '--type', dest='type',
                    action='store', default="exec")
a = parser.parse_args()

if a.type not in LOG_TYPES:
    raise Exception("Please specify a valid log type")

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
    size += ((previous_dimension * DIGITS) + DIGITS)
    return size

def parseCorrect(lentry):
    '''
    Expects a string like "Network classified 7971 (10000) images correctly" or "Epoch: 10 | Loss: 584.342| Correct: 5430"
    Returns first 4 digit value starting with a space
        > 7971 or 5430
    '''
    c_match = re.search(r' \d\d\d\d', lentry)
    if c_match == None:
        raise Exception("Could not find total correct classification", lentry)
    return int(c_match.group(0))

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
        raise Exception("Unknown program found : " + command)
    return prog, o[0].epochs, o[0].shape

if (a.type == LOG):
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
elif (a.type == ALOG):
    for entry in [ log[i : i + ALOG_ENTRY_SIZE]
              for i in range(0, len(log), ALOG_ENTRY_SIZE)]:
        if (len(entry) == ALOG_ENTRY_SIZE):
            # Checkout sample_alog_entry to understand the numbers here
            p, e, s = parseCommand(entry[0][18:])
            corrects = [parseCorrect(e) for e in entry[1:]]
            results[p].append({
                "epochs"  : e,
                "shape"   : s,
                "size"    : getSize(s),
                "corrects" : corrects
            })

print(json.dumps(results, sort_keys=True, indent=4))
