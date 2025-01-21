#include <cstdio>
#include <cuda_runtime.h>

// 需要启用分离声明和定义编译，不建议分离，方便编译器优化
__device__ void say_hello();

__global__ void another() {
    printf("another: Thread %d of %d\n", threadIdx.x, blockDim.x);
}

__global__ void kernel() {
    // CUDA特殊变量，只有核函数可以获取
    // threadIdx.x 线程编号
    // blockDim.x 线程数量
    // blockIdx.x 板块编号
    // gridDim.x 板块数量
    printf("Block %d of %d. Thread %d of %d\n", 
            blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tnum = blockDim.x * gridDim.x;
    printf("Flattened Thread %d of %d\n", tid, tnum);
    say_hello();
    // 核函数调用核函数
    int numthreads = threadIdx.x * threadIdx.x + 1;
    another<<<1, numthreads>>>();
    printf("kernel: called another with %d threads\n", numthreads);
}

int main() {
    // 第一个参数决定启动核函数时的板块数
    // 第二个参数决定启动核函数时的线程数
    // 板块和线程之间都是并行的，不一定能保证顺序
    // <<<x, y>>>即创建一个grid，grid有x个block，每个block有y个thread
    kernel<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}