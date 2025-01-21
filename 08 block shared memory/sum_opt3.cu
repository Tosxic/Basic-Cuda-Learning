#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "ticktock.h"


// 使用板块和板块共享内存实现并行reduce
__global__ void parallel_sum(int *sum, int const *arr, int n) {
    // 板块共享内存
    __shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;
    // SM执行一个板块的线程时不是全部同时执行的，是32个为一组
    // 0~31为一个warp, 32~63为一个warp
    // 当某组线程陷入内存等待，则切换至其他组线程，因此会出现顺序问题，导致结果出错
    // 通过__syncthreads()指令强制同步所有线程，板块内所有线程运行至指令才会继续运行
    // 因为32个线程一组，所以j<32不需要强制同步
    // 但需要对共享内存进行volatile禁止编译优化
    local_sum[j] = arr[i*1024+j];
    __syncthreads();
    if(j < 512) {
        local_sum[j] += local_sum[j+512];
    }
    __syncthreads();
    if(j < 256) {
        local_sum[j] += local_sum[j+256];
    }
    __syncthreads();
    if(j < 128) {
        local_sum[j] += local_sum[j+128];
    }
    __syncthreads();
    if(j < 64) {
        local_sum[j] += local_sum[j+64];
    }
    __syncthreads();
    if(j < 32) {
        local_sum[j] += local_sum[j+32];
    }
    if(j < 16) {
        local_sum[j] += local_sum[j+16];
    }
    if(j < 8) {
        local_sum[j] += local_sum[j+8];
    }
    if(j < 4) {
        local_sum[j] += local_sum[j+4];
    }
    if(j < 2) {
        local_sum[j] += local_sum[j+2];
    }
    if(j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}

int main() {
    int n = 1 << 24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 1024);

    for(int i = 0; i < n; ++i) {
        arr[i] = i % 4;
    }

    TICK(parallel_sum);
    parallel_sum<<<n / 1024, 1024>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());

    int final_sum = 0;
    for(int i = 0; i < n / 1024; ++i) {
        final_sum += sum[i];
    }
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);
    return 0;
}