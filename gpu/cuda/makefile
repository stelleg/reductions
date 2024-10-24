targets := reduce_shfl reduce_shared reduce_shfl_uvm reduce_cub reduce_naive
all: $(targets)

reduce_shfl: reduce_shfl.cu
	nvcc $< -O2 -o $@
	
reduce_cub: reduce_cub.cu
	nvcc $< -O2 -o $@

reduce_shfl_uvm: reduce_shfl_uvm.cu
	nvcc $< -O2 -o $@

reduce_naive: reduce_naive.cu
	nvcc $< -O2 -o $@

reduce_shared: reduce_shared.cu
	nvcc $< -O2 -o $@

clean: 
	rm 	-f $(targets)
