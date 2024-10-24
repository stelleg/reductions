cxx=clang++
KOKKOS_DEVICES=OpenMP

all: reduce reduce-gpu reduce-opt.ll

reduce: main.cc reductions.h magma.h
	${cxx} -Wall main.cc -o reduce -O2 -ftapir=opencilk -fopenmp -ldl

reduce-opt.ll: main.cc reductions.h magma.h
	${cxx} -O2 -S -emit-llvm -ftapir=none -fopenmp -o $@ $<

reduce-opt-cilk.ll: main.cc reductions.h magma.h
	${cxx} -O2 -S -emit-llvm -ftapir=openmp -o $@ $<

reduce-gpu:
	${cxx} -O2 main.cc -o reduce-gpu -ftapir=gpu -lllvm-gpu -fno-exceptions
