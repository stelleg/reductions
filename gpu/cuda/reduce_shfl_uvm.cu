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
__inline__ __device__
a warpReduce(a in){
  for(int offset = warpSize / 2; offset > 0; offset /= 2)
    in = op(in, __shfl_down_sync(0xffffffff, in, offset)); 
  return in; 
}

// requires blocksize = warpSize²
template <typename a>
__inline__ __device__
a blockReduce(a acc){
  __shared__ a shared[warpSize]; 
  // Block reduce
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;
  
  acc = warpReduce(acc);
  if(lane == 0) shared[warp] = acc; 
  __syncthreads();

  // Only the first warp does the final reduction for the block 
  if(warp == 0) acc = warpReduce(shared[lane]);
    
  return acc; 
}
  

// Requires blockDim = gridDim = warpSize²
template <typename a>
__global__ 
void gpuReduce(a *in, a* out, uint64_t n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nthr = blockDim.x * gridDim.x; 
  if(i < n){
    a acc = in[i];
    for(i+= nthr; i < n; i+= nthr)
      acc = op(acc, in[i]); 

    acc = blockReduce(acc); 
    if(threadIdx.x == 0) out[blockIdx.x] = acc; 
  }
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
  double * xs; cudaMallocManaged(&xs, n*sizeof(double)); 

  gpuInit<double><<<256, 1024>>>(xs, n); 
  checkCUDA(cudaGetLastError()); 
  checkCUDA(cudaDeviceSynchronize()); 

  // Sequential reduction
  double seq_red = 0.0; 
  for(uint64_t i=0; i<n; i++){
    seq_red += xs[i];
  }
  printf("seq red: %f\n", seq_red);
  printf("warpSize: %d\n", warpSize); 
  int blockdim = warpSize * warpSize;
  int griddim = blockdim;

  double* out = (double*)malloc(griddim*sizeof(double));

  // Warm up
  gpuReduce<double><<<griddim, blockdim>>>(xs, out, n); 
  checkCUDA(cudaGetLastError()); 
  checkCUDA(cudaDeviceSynchronize()); 
  gpuReduce<double><<<1, griddim>>>(out, out, griddim); 
  checkCUDA(cudaGetLastError()); 
  checkCUDA(cudaDeviceSynchronize()); 
  printf("red: %f\n", out[0]); 
    
  clock_t before = clock(); 
  int niter = 100; 
  for(int it = 0; it < niter; it++){
    gpuReduce<double><<<griddim, blockdim>>>(xs, out, n); 
    cudaDeviceSynchronize(); 
    gpuReduce<double><<<1, griddim>>>(out, out, griddim); 
    cudaDeviceSynchronize(); 
  }
  clock_t after = clock();

  double duration = (double)(after - before) / 1000000; 
  double gb = ((double)n * niter * sizeof(double)) / 1000000000 ; 
  printf("Test duration %.4fs\n", duration); 
  printf("Reduction throughput = %.4f GB/s\n", gb /duration); 

  //free(xs);
  cudaFree(xs); 
}
