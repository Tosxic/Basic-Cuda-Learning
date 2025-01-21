#include <cstdio>
#include <cuda_runtime.h>

// __device__修饰，从GPU调用，可以有参数和返回值
// __inline__语义即内联优化，不含weak符号语义，不保证内联
// __forceinline__强制内联
// __noinline__禁止内联优化
__device__ __inline__ void say_hello() {
    printf("Hello, world from GPU!\n");
}

// __host__修饰，从CPU调用，可以省略，可以有参数和返回值
__host__ void say_hello_host() {
    printf("Hello, world from CPU!\n");
}

// __host__和__device__可以联合使用，从CPU和GPU都可以调用
__host__ __device__ void say_hello_all() {
// cuda编译先通过cpu编译器生成CPU指令，再送到GPU编译器生成GPU指令，最后链接
// 在GPU编译模式下，会定义__CUDA_ARCH__宏
#ifdef __CUDA_ARCH__
    printf("Hello, world from GPU architecture %d!\n", __CUDA_ARCH__);
#else
    printf("Hello, world from CPU!\n");
#endif
}

// 针对不同架构使用不同代码
__host__ __device__ void func() {
#if __CUDA_ARCH__ >= 700
    // xxx
#elif __CUDA_ARCH__ >= 600
    // xxx
#elif __CUDA_ARCH__ >= 500
    // xxx
#elif __CUDA_ARCH__ >= 300
    // xxx
#elif !defined(__CUDA_ARCH__)
    // xxx
#endif
}

// __global__修饰，可以从CPU调用，在GPU上执行，必须是void，可以有参数
__global__ void kernel() {
    // say_hello();
    say_hello_all();
}

//main()运行在CPU
int main() {
    kernel<<<1, 1>>>();
    // CPU和GPU间通信是异步的，可以用cudaDeviceSynchronize()等待GPU执行完
    cudaDeviceSynchronize();
    // say_hello_host();
    say_hello_all();
    return 0;
}