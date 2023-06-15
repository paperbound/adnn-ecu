# benchmark hdrnn

# tools
LOG=$(date '+hlog.%C%y.%m.%d.%H')

# hdrnn
declare -a PROGRAMS=(
	"c-math.h"
	"cpp-libtorch"
	"cpp-eigen"
	"python-numpy"
)

declare -a PROGRAM_CMDS=(
	"./hdrnn/c-math.h/bin/hdrnn train --epochs 3 --quiet --shape 32"
	"./hdrnn/cpp-libtorch/build/mnist train --epochs 3 --quiet --shape 32"
	"./hdrnn/cpp-eigen/bin/hdr train --epochs 3 --quiet --shape "
	"python3 ./hdrnn/python-numpy/train.py --epochs 3 --quiet --shape 32"
)

{
	for ((idx=0; idx<${#PROGRAMS[@]}; ++idx))
	do
		heaptrack -o "${PROGRAMS[idx]}.heaptrack" -- ${PROGRAM_CMDS[idx]}
	done
} >> ${LOG} 2>&1
