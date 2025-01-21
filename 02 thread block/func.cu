#include <cstdio>
#include <cuda_runtime.h>

__device__ void say_hello() {
    printf("Hello, world!\n");
}