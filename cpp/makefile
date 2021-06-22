cxx=~/llvm-project/build/bin/clang++
KOKKOS_DEVICES=OpenMP

reduce: main.cc reductions.h magma.h
	${cxx} -Wall main.cc -o reduce -O2 -ftapir=cilk -fopenmp -lkokkoscore -ldl

reduce-opt.ll: main.cc reductions.h magma.h
	${cxx} -O2 -S -emit-llvm -ftapir=none -fopenmp -o $@ $<

reduce-opt-cilk.ll: main.cc reductions.h magma.h
	${cxx} -O2 -S -emit-llvm -ftapir=openmp -o $@ $<
