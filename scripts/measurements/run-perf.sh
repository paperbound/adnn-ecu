# benchmark hdrnn

# tools
LOG=$(date '+plog.%C%y.%m.%d.%H')
PERF_RECORD="perf record -F 99 -g --"
PERF_SCRIPT="perf script"

# hdrnn
declare -a PROGRAMS=(
	"c-math.h"
	"cpp-libtorch"
	"cpp-eigen"
	"python-numpy"
)

declare -a PROGRAM_CMDS=(
	"./hdrnn/c-math.h/bin/hdrnn train --epochs 1 --quiet --shape 32"
	"./hdrnn/cpp-libtorch/build/mnist train --epochs 1 --quiet --shape 32"
	"./hdrnn/cpp-eigen/bin/hdr train --epochs 1 --quiet --shape "
	"python3 ./hdrnn/python-numpy/train.py --epochs 1 --quiet --shape 32"
)

{
	for ((idx=0; idx<${#PROGRAMS[@]}; ++idx))
	do
		${PERF_RECORD} ${PROGRAM_CMDS[idx]}
		${PERF_SCRIPT} > "${PROGRAMS[idx]}.perf-script"
	done
} >> ${LOG} 2>&1
