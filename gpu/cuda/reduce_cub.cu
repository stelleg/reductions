#include<stdio.h>
#include<time.h>
#include<cuda_runtime.h>
#include<iostream>
#include<stdint.h>
#include<assert.h>
#include<cub/cub.cuh>

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

template <typename a> 
__inline__ __device__  
void  reduce(a* x, a y){
  *x = *x + y;
}

void *temps = nullptr;
size_t temps_size = 0;

template<typename a> 
a gpuReduce(a *ins, a* outs, size_t n) {
    // CUB can't handle large arrays with over 2 billion elements!
    assert(n < std::numeric_limits<int>::max());

    a res; 
    // Determine temporary device storage requirements
    if(!temps) {
      checkCUDA(cub::DeviceReduce::Sum(temps, temps_size, ins, outs, n)); 
      printf("allocating %d bytes for cub\n", temps_size); 
      checkCUDA(cudaMalloc(&temps, temps_size)); 
    }
    checkCUDA(cub::DeviceReduce::Sum(temps, temps_size, ins, outs, n));
    checkCUDA(cudaMemcpy(&res, outs, sizeof(double), cudaMemcpyDeviceToHost)); 

    return res;  
}

template <typename a>
__global__ void gpuInit(a* xs, uint64_t n){
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + tid; 
  unsigned int gridSize = blockDim.x*gridDim.x;
  extern __shared__ a sdata[]; 
  while(i<n) {
    xs[i] = (a)i; 
    i+=gridSize; 
  }
}

int main(int argc, char** argv){
  uint64_t n = 1ULL << (argc > 1 ? atoi(argv[1]) : 28);
  printf("n = %ld\n", n); 
  double* xs;  checkCUDA(cudaMalloc(&xs, n*sizeof(double))); 
  double* out; checkCUDA(cudaMalloc(&out, sizeof(double))); 
  constexpr uint32_t dimBlock = 1UL << 8; 
  printf("blocksize = %d\n", dimBlock); 
  constexpr uint64_t dimGrid = 1UL << 8; 
  printf("gridsize = %ld\n", dimGrid); 
  double* reds; checkCUDA(cudaMalloc(&reds, dimGrid*sizeof(double))); 

  gpuInit<double><<<dimGrid, dimBlock>>>(xs, n); 
  checkCUDA(cudaDeviceSynchronize()); 
  // Warm up
  double res = gpuReduce<double>(xs, out, n); 
  printf("red: %f\n", res); 
    
  clock_t before = clock(); 
  int niter = 1000; 
  for(int it = 0; it < niter; it++){
    double res = gpuReduce<double>(xs, out, n); 
  }
  clock_t after = clock();

  double duration = (double)(after - before) / 1000000; 
  double gb = ((double)n * niter * sizeof(double)) / 1000000000 ; 
  printf("Test duration %.4fs\n", duration); 
  printf("Reduction throughput = %.4f GB/s\n", gb /duration); 

  checkCUDA(cudaFree(xs));
  checkCUDA(cudaFree(reds));
}
