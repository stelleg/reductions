all: reduce_shfl reduce_shared

reduce_shfl: reduce_shfl.cu
	nvcc $< -O2 -o $@

reduce_shared: reduce_shared.cu
	nvcc $< -O2 -o $@
