# benchmark hdrnn

# tools
LOG=$(date '+log.%C%y.%m.%d.%H')
TIME="/usr/bin/time -v --"

# hdrnn
declare -a PROGRAMS=(
	"./hdrnn/c-math.h/bin/hdrnn train --epochs 1 --quiet --shape "
	"./hdrnn/cpp-libtorch/build/mnist train --epochs 1 --quiet --shape "
	"./hdrnn/cpp-eigen/bin/hdr train --epochs 1 --quiet --shape "
	"python3 ./hdrnn/python-numpy/train.py --epochs 1 --quiet --shape "
)
declare -a SIZES=("2" "4" "8" "16,16" "32" "48,48" "64,16" "72" "82,36,16" "96,96" "104" "114" "128")

{
	for program in "${PROGRAMS[@]}"
	do
		for size in "${SIZES[@]}"
		do
			${TIME} ${program} ${size}
		done
	done
} >> ${LOG} 2>&1
