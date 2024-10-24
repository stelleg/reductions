#define __STRICT_ANSI__ 1
#include<stdio.h>
#include<time.h>
#include<cuda_runtime.h>
#include<iostream>
#include<cstdint>

#define checkCUDA(condition)                                                                \
    {                                                                                       \
        const cudaError_t error = condition;                                                 \
        if(error != cudaSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << cudaGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error);                                                     \
        }                                                                                   \
    }


#define warpSize 32

template <typename a> 
__inline__ __device__  
a  op(a x, a y){
  return x + y;
}

template <typename a>
__global__ 
void gpuReduce(a *in, a* out, uint64_t n){
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t nthr = blockDim.x * gridDim.x; 
  a acc; 
  uint64_t i = tid; 
  if(i < n) acc = in[i];  
  for(i += nthr; i < n; i+= nthr){
    acc = op(acc, in[i]); 
  }
  out[tid] = acc; 
}

template <typename a>
__global__ 
void gpuReduce2(a *in, a* out, uint64_t n){
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t nthr = blockDim.x * gridDim.x; 
  uint64_t k = n / nthr + (n % nthr > 0); 
  uint64_t i = tid * k; 
  a acc; 
  if(i < n) acc = in[i];  
  for(i++; i < (tid+1)*k; i++){
    acc = op(acc, in[i]); 
  }
  out[tid] = acc; 
}

template <typename a>
__global__ void gpuInit(a* xs, uint64_t n){
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + tid; 
  unsigned int gridSize = blockDim.x*gridDim.x;
  while(i<n) {
    xs[i] = (a)i; 
    i+=gridSize; 
  }
}

int main(int argc, char** argv){
  uint64_t n = 1ULL << (argc > 1 ? atoi(argv[1]) : 28);
  printf("n = %ld\n", n); 
  //double* xs = (double*)malloc(n*sizeof(double)); 
  double* xs; cudaMallocManaged(&xs, n*sizeof(double)); 

  gpuInit<double><<<256, 1024>>>(xs, n); 
  checkCUDA(cudaGetLastError()); 
  checkCUDA(cudaDeviceSynchronize()); 

  // Sequential reduction
  double seq_red = 0.0; 
  for(uint64_t i=0; i<n; i++){
    seq_red += xs[i];
  }

  int devId; 
  checkCUDA(cudaGetDevice(&devId)); 
  //checkCUDA(cudaMemPrefetchAsync(xs, n*sizeof(double), devId)); 
  printf("seq red: %f\n", seq_red);
  printf("warpSize: %d\n", warpSize); 
  int blockdim = warpSize * warpSize;
  int griddim = blockdim;
  int nthr = griddim*blockdim; 

  double* out = (double*)malloc(nthr*sizeof(double));

  // Warm up
  gpuReduce<double><<<griddim, blockdim>>>(xs, out, n); 
  checkCUDA(cudaGetLastError()); 
  checkCUDA(cudaDeviceSynchronize()); 
  double red = 0; 
  for(int i=0; i<nthr; i++){
    red += out[i]; 
  }
  printf("red: %f\n", red); 
    
  clock_t before = clock(); 
  int niter = 1000; 
  for(int it = 0; it < niter; it++){
    gpuReduce<double><<<griddim, blockdim>>>(xs, out, n); 
    checkCUDA(cudaDeviceSynchronize()); 
    double acc = 0; 
    for(int i=0; i<nthr; i++){
      acc += out[i]; 
    }
  }
  clock_t after = clock();

  double duration = (double)(after - before) / 1000000; 
  double gb = ((double)n * niter * sizeof(double)) / 1000000000 ; 

  printf("Stride = gridsize\n"); 
  printf("Test duration %.4fs\n", duration); 
  printf("Reduction throughput = %.4f GB/s\n", gb /duration); 

  before = clock(); 
  for(int it = 0; it < niter; it++){
    gpuReduce2<double><<<griddim, blockdim>>>(xs, out, n); 
    checkCUDA(cudaDeviceSynchronize()); 
    double acc = 0; 
    for(int i=0; i<nthr; i++){
      acc += out[i]; 
    }
  }
  after = clock();

  duration = (double)(after - before) / 1000000; 
  gb = ((double)n * niter * sizeof(double)) / 1000000000 ; 

  printf("Stride = 1\n"); 
  printf("Test duration %.4fs\n", duration); 
  printf("Reduction throughput = %.4f GB/s\n", gb /duration); 

  //free(xs); 
  cudaFree(xs);
}
