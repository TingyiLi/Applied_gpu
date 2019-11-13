#include <stdio.h>
#define TPB 64
#define numthread 256

__global__ void HelloWorld()
{
    const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Hello World! My threadId is %d\n", gridIdx.x);
    //printf("threadID %d, blockIdx %d, blockDim %d\n",threadID, blockIdx.x, blockDim.x);
}

int main()
{
    HelloWorld<<<1, numthread>>>();
    cudaDeviceSynchronize();
    return 0;
}
