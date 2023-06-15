# benchmark hdrnn

# tools
LOG=$(date '+alog.%C%y.%m.%d.%H')

# hdrnn
declare -a PROGRAMS=(
	"./hdrnn/c-math.h/bin/hdrnn train --epochs 11 --shape "
	"./hdrnn/cpp-libtorch/build/mnist train --epochs 11 --shape "
	"./hdrnn/cpp-eigen/bin/hdr train --epochs 11 --shape "
	"python3 ./hdrnn/python-numpy/train.py --epochs 11 --shape "
)
declare -a SIZES=("2" "4" "8" "16,16" "16,32,16" "32" "32,32,32" "48" "64,16" "72" "96" "104" "114" "128")

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
