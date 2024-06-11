#include<stdio.h>
#include<time.h>
#include<hip/hip_runtime.h>
#include<iostream>
#include<stdint.h>

#define checkHIP(i) (assert((i) == hipSuccess))

template <typename a> 
__inline__ __device__  
void  reduce(a* x, a y){
  *x = *x + y;
}

template <typename a, uint32_t blockSize, uint32_t numStrides>
__global__ void gpuReduce(a* xs, a* localreds, a unit, uint64_t n){
  extern __shared__ a sdata[]; 
  uint32_t tid = threadIdx.x;
  uint64_t i = blockIdx.x*(blockSize*numStrides) + tid; 
  uint32_t gridSize = blockSize*numStrides*gridDim.x;
  sdata[tid] = unit;
  while(i<n) { 
    for(int j=0; j<numStrides; j++){
      reduce(sdata+tid, xs[i+j*blockSize]); 
    }
    i+= gridSize; 
  } 
  __syncthreads();

  for(int i=512; i>1; i >>= 1){
    if(blockSize >= i){
      if(tid < max(i/2,32)){
        reduce(sdata+tid, sdata[tid+i/2]); 
        __syncthreads(); 
      }
    }
  }

  if(tid == 0) localreds[blockIdx.x] = sdata[tid]; 
}

template <typename a>
__global__ void gpuInit(a* xs, uint64_t n){
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + tid; 
  unsigned int gridSize = blockDim.x*gridDim.x;
  extern __shared__ a sdata[]; 
  while(i<n) {
    xs[i] = 1.00; 
    i+=gridSize; 
  }
}

int main(){
  uint64_t n = 1ULL << 28;
  printf("n = %ld\n", n); 
  double* xs;  checkHIP(hipMalloc(&xs, n*sizeof(double))); 
  double* host_xs = (double*)malloc(n*sizeof(double)); 
  constexpr uint32_t dimBlock = 1UL << 8; 
  printf("blocksize = %d\n", dimBlock); 
  constexpr uint64_t dimGrid = 1UL << 8; 
  printf("gridsize = %ld\n", dimGrid); 
  double* reds; checkHIP(hipMalloc(&reds, dimGrid*sizeof(double))); 
  double* host_reds = (double*)malloc(dimGrid*sizeof(double)); 

  gpuInit<double><<<dimGrid, dimBlock>>>(xs, n); 
  checkHIP(hipMemcpy(host_xs, xs, n*sizeof(double), hipMemcpyDeviceToHost)); 

  // Sequential reduction
  double seq_red = 0.0; 
  for(uint64_t i=0; i<n; i++){
    seq_red += host_xs[i];
  }
  printf("seq red: %f\n", seq_red);

  // Warm up
  gpuReduce<double, dimBlock, 2><<<dimGrid, dimBlock, 2*dimBlock*sizeof(double)>>>(xs, reds, 0.0, n); 

  checkHIP(hipMemcpy(host_reds, reds, dimGrid*sizeof(double), hipMemcpyDeviceToHost));

  double red = 0.0; 
  for(int i=0; i<dimGrid; i++){
    red += host_reds[i]; 
  }
  printf("red: %f\n", red); 
    
  clock_t before = clock(); 
  int niter = 100; 
  for(int it = 0; it < niter; it++){
    gpuReduce<double, dimBlock, 2><<<dimGrid, dimBlock, 2*dimBlock*sizeof(double)>>>(xs, reds, 0.0, n); 
    checkHIP(hipMemcpy(host_reds, reds, dimGrid*sizeof(double), hipMemcpyDeviceToHost)); 
    double red = 0.0; 
    for(int i=0; i<dimGrid; i++){
      // expected local variable to be 
      double elv = (double)n / dimGrid; 
      //if(abs(reds[i] - elv) > 0.1) 
      //printf("%f, %f \n", reds[i], elv); 
      red += host_reds[i]; 
    }
  }
  clock_t after = clock();

  double duration = (double)(after - before) / 1000000; 
  double gb = ((double)n * niter * sizeof(double)) / 1000000000 ; 
  printf("Test duration %.4fs\n", duration); 
  printf("Reduction throughput = %.4f GB/s\n", gb /duration); 

  checkHIP(hipFree(xs));
  checkHIP(hipFree(reds));
  free(host_xs);
  free(host_reds); 
}
