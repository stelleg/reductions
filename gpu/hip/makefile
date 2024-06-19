all: reduce_shfl reduce_shared

reduce_shfl: reduce_shfl.cpp
	hipcc $< -O2 -o $@

reduce_shared: reduce_shared.cpp
	hipcc $< -O2 -o $@
