#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

// 由于是异步的，所以核函数返回时结果并未运算完成，因此无返回值
// 传递CPU栈\堆指针来获取结果也无法获取结果，因为cpu内存和显存独立
// 同样的，使用cudaMalloc申请显存可以在核函数上使用，但cpu不能访问
__global__ void kernel(int *pret) {
    *pret = 42;
}

// 并行赋值
__global__ void get_array(int *arr, int n) {
    int i = threadIdx.x;
    arr[i] = i;
}

// 线程数受限，并行赋值
__global__ void get_array_limited(int *arr, int n) {
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        arr[i] = i;
    }
}

// 线程数受限，通过block并行赋值
__global__ void get_array_block(int *arr, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i >= n) {
        return ;
    }
    arr[i] = i;
}

int main() {
    int ret = 0;
    int *pret;

    // kernel<<<1, 1>>>(&ret);
    // cuda报错不在命令行立马回显，会静默返回错误码
    // cudaError_t err = cudaDeviceSynchronize();
    // printf("error code: %d\n", err);
    // printf("error name: %s\n", cudaGetErrorName(err));
    // 可以使用cuda samples里的helper_cuda.h来回显
    // checkCudaErrors(cudaDeviceSynchronize());
    // printf("%d\n", ret);

    // cudaMalloc申请显存
    // checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
    // kernel<<<1, 1>>>(pret);
    // checkCudaErrors(cudaDeviceSynchronize());
    // cudaMemcpy跨设备内存拷贝，且隐含设备同步操作，因此上面的cudaDeviceSynchronize不需要
    // checkCudaErrors(cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost));
    // cudaFree(pret);
    // printf("%d\n", ret);

    // cudaMallocManaged采用统一内存地址技术(较新显卡特性)，有一定开销成本
    // 即分配的地址在GPU和CPU是相同的，都可以访问，拷贝也是按需进行，但不包含同步操作
    checkCudaErrors(cudaMallocManaged(&pret, sizeof(int)));
    kernel<<<1, 1>>>(pret);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("result: %d\n", *pret);
    cudaFree(pret);

    int n = 32;
    int *arr;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    get_array<<<1, n>>>(arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    for(int i = 0; i < n; ++i) {
        printf("arr[%d]: %d\n", i, arr[i]);
    }
    cudaFree(arr);

    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    get_array_limited<<<1, 4>>>(arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    for(int i = 0; i < n; ++i) {
        printf("arr[%d]: %d\n", i, arr[i]);
    }
    cudaFree(arr);

    n = 65535;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    int nthreads = 128;
    int nblocks = (n + nthreads - 1) / nthreads;
    get_array_block<<<nblocks, nthreads>>>(arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    for(int i = 0; i < n; ++i) {
        printf("arr[%d]: %d\n", i, arr[i]);
    }
    cudaFree(arr);
    return 0;
}