#include<Kokkos_Core.hpp>
#include<Kokkos_Vector.hpp>
#include<vector>
#include<array>
#include<iostream>
#include<string>
#include<chrono>
#include"reductions.h"

using namespace std; 

int main(){
  // integer sums, products, and all max/min: commutative and assocative
  vector<int> xs = {1,2,3,4,5,6}; 
  cout << "sum: " << reduce<int>(Sum<int>(), xs) << endl; 
  cout << "par sum: " << parReduce<int>(Sum<int>(), xs, 2) << endl; 
  cout << "max: " << reduce<int>(Max<int>(), xs) << endl; 

  // string concatenation: associative, not commutative
  vector<string> ss = {"hello ", "world"}; 
  cout << "string append: " << reduce<string>(StringApp(), ss) << endl;

  // float sums and products: commutative, not associative
  vector<float> ys = {3.14, 2.71}; 
  cout << "float sum: " << reduce<float>(Sum<float>(), ys) << endl; 
  cout << "float product: " << reduce<float>(Product<float>(), ys) << endl; 

  // here we compare performance of sequential vs parallel reductions
  uint64_t n = 1ull<<31; 
  vector<double> big(n, 3.14); 

  //sequential 
  auto start = chrono::high_resolution_clock::now(); 
  double sum = reduce<double>(Sum<double>(), big); 
  auto stop = chrono::high_resolution_clock::now(); 
  cout << "seq reduce: " << sum << ", " << chrono::duration_cast<chrono::milliseconds>(stop-start).count() << " ms" << endl; 

  //parallel
  start = chrono::high_resolution_clock::now(); 
  sum = parReduce<double>(Sum<double>(), big, 64);  
  stop = chrono::high_resolution_clock::now(); 
  cout << "par reduce: " << sum << ", " << chrono::duration_cast<chrono::milliseconds>(stop-start).count() << " ms" << endl; 

  //openmp
  sum = 0;
  start = chrono::high_resolution_clock::now(); 
  #pragma omp parallel for reduction(+:sum)
  for(uint64_t i=0; i<n; i++){
    sum += big[i];
  }
  stop = chrono::high_resolution_clock::now(); 
  cout << "omp reduce: "  << sum << ", " << chrono::duration_cast<chrono::milliseconds>(stop-start).count() << " ms" << endl; 

  //kokkos
  double sum = 0;
  auto start = chrono::high_resolution_clock::now(); 
  Kokkos::parallel_reduce(n, [&big](const uint64_t i, double& y){
    y += big[i];
  }, [=](double& dest, const double&src){ dest += src; }); 
  auto stop = chrono::high_resolution_clock::now(); 
  cout << "kokkos reduce: "  << sum << ", " << chrono::duration_cast<chrono::milliseconds>(stop-start).count() << " ms" << endl; 
}
