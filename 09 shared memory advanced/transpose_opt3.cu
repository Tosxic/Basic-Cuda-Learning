#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "ticktock.h"

// 通过共享内存（二级缓存）降低主存跨步开销
// template<int blockSize, class T>
// __global__ void parallel_transpose(T *out, T const *in, int nx, int ny) {
//     int x = blockIdx.x * blockSize + threadIdx.x;
//     int y = blockIdx.y * blockSize + threadIdx.y;
//     if(x >= nx || y >= ny) {
//         return ;
//     }
//     __shared__ T tmp[blockSize*blockSize];
//     int rx = blockIdx.y * blockSize + threadIdx.x;
//     int ry = blockIdx.x * blockSize + threadIdx.y;
//     tmp[threadIdx.y*blockSize+threadIdx.x] = in[ry*nx+rx];
//     __syncthreads();
//     out[y*nx+x] = tmp[threadIdx.x*blockSize+threadIdx.y];
// }

// 共享内存由32个区块（bank）并联组成，为网格状，addr存储在bank[addr%32]中
// 当并行访问时，会发生区块冲突，即多个线程同时访问一个bank会变成串行
// 因此，上述程序可以在使用共享内存时不对齐，跨步改为33，防止区块冲突，进一步提升性能
template<int blockSize, class T>
__global__ void parallel_transpose(T *out, T const *in, int nx, int ny) {
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;
    if(x >= nx || y >= ny) {
        return ;
    }
    __shared__ T tmp[(blockSize+1)*blockSize];
    int rx = blockIdx.y * blockSize + threadIdx.x;
    int ry = blockIdx.x * blockSize + threadIdx.y;
    tmp[threadIdx.y*(blockSize+1)+threadIdx.x] = in[ry*nx+rx];
    __syncthreads();
    out[y*nx+x] = tmp[threadIdx.x*(blockSize+1)+threadIdx.y];
}


int main() {
    int nx = 1 << 12, ny = 1 << 12;
    std::vector<int, CudaAllocator<int>> in(nx * ny);
    std::vector<int, CudaAllocator<int>> out(nx * ny);

    for(int i = 0; i < nx * ny; ++i) {
        in[i] = i;
    }

    TICK(parallel_transpose);
    parallel_transpose<32><<<dim3(nx / 32, ny / 32, 1), dim3(32, 32, 1)>>>
        (out.data(), in.data(), nx, ny);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_transpose);

    for(int y = 0; y < ny; ++y) {
        for(int x = 0; x < nx; ++x) {
            if(out[y*nx+x] != in[x*nx+y]) {
                printf("Wrong At x=%d, y=%d: %d != %d\n", x, y,
                        out[y*nx+x], in[x*nx+y]);
                return -1;
            }
        }
    }

    printf("All Correct!\n");
}