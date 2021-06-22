#include <kitsune.h>

int main(int argc, char**argv){
  int n = argc > 1 ? atoi(argv[1]) : 10; 
  float acc = 0; 
  forall(int i=0; i<n; i++){
    acc += i*3.1415; 
  }
  printf("%f\n", acc);
}

