# benchmark hdrnn

# tools
LOG=$(date '+alog.%C%y.%m.%d.%H')

# hdrnn
declare -a PROGRAMS=(
	"./hdrnn/c-math.h/bin/hdrnn train --epochs 14 --shape "
	"./hdrnn/cpp-libtorch/build/mnist train --epochs 14 --shape "
	"./hdrnn/cpp-eigen/bin/hdr train --epochs 14 --shape "
	"python3 ./hdrnn/python-numpy/train.py --epochs 14 --shape "
)
declare -a SIZES=("2" "4" "8" "16,16" "32" "32,32,32" "64" "96" "128")

{
	for program in "${PROGRAMS[@]}"
	do
		for size in "${SIZES[@]}"
		do
			echo "Running command: ${program} ${size}"
			${program} ${size}
		done
	done
} >> ${LOG} 2>&1
