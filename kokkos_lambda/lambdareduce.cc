#include<Kokkos_Core.hpp>

namespace Kokkos {

template <typename I, typename Acc, typename Body, typename Join, Body b, Join j>
struct LambdaFunctor {
  using value_type = Acc; 
  using size_type = I; 
  
  template<typename ... Args>
  KOKKOS_INLINE_FUNCTION
  auto join(Args... as) const -> 
    decltype(j(as...)){
      return j(as...);
  }; 
  template<typename ... Args>
  KOKKOS_INLINE_FUNCTION
  auto operator()(Args... as) const ->
    decltype(b(as...)) {
      return b(as...); 
  };
}; 

template <typename I, typename Body, typename Join, typename Acc>
auto parallel_reduce(I n, Body b, Join j, Acc& a){
  return parallel_reduce(n, LambdaFunctor<I, Acc, Body, Join, b, j>(), a);  
}

}

int main(int argc, char** argv){
  Kokkos::initialize(argc, argv); 
  auto body = [](size_t i, int& j){ printf("body %lu %d\n", i, j); j += i; }; 
  auto join = [](int& dst, const int& src){ printf("join %d %d\n", dst, src); dst += src; }; 
  int s;
  s = 0; 
  Kokkos::parallel_reduce(20, body, join, s); 
  printf("result: %d\n", s); 
  s = 0; 
  Kokkos::parallel_reduce(20, body, s); 
  printf("result: %d\n", s); 
}
