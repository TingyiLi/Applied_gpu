#include <stdio.h>
#include <sys/time.h>
#define ARRAY_SIZE 1000000
#define blocksize 256
#define TRUE 1
#define FALSE 0

void cpu_saxpy(int n, float a, float *x, float *y)
{
	for(int i = 0; i < n; i++)
		y[i] = a*x[i] + y[i];
}

__global__ void gpu_saxpy(int n, float a, float *x, float *y)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n) y[i] = a*x[i] + y[i];
}

int cmp_saxpy(int n, float *cpu_y, float *gpu_y)
{
	const float round_err = 0.0001;
	for(int i = 0; i < ARRAY_SIZE ; i++)
	{
		if(fabs(cpu_y[i] - gpu_y[i]) >= round_err)
			return FALSE;
	}

	return TRUE;
}

double timeeval(struct timeval t0, struct timeval t1)
{
	return (double)(t1.tv_sec - t0.tv_sec) * 1000.0L + (double)(t1.tv_usec - t0.tv_usec) / 1000.0L;
}


int main()
{

	srand((unsigned int)time(NULL));
	float MAX_GEN = 50.0;

	float *x_gpu, *y_gpu, *x_cpu, *y_cpu;
  // float *x_gpu, *y_gpu;

	cudaMalloc(&x_gpu, ARRAY_SIZE*sizeof(float));
	cudaMalloc(&y_gpu, ARRAY_SIZE*sizeof(float));
	float y_cpu_res[ARRAY_SIZE];
  x_cpu = (float*)malloc(ARRAY_SIZE*sizeof(float));
  y_cpu = (float*)malloc(ARRAY_SIZE*sizeof(float));

	for(int i = 0; i < ARRAY_SIZE; i++)
	{
		x_cpu[i] = (float)rand()/(float)(RAND_MAX/MAX_GEN);
		y_cpu[i] = (float)rand()/(float)(RAND_MAX/MAX_GEN);
	}

  struct timeval start, end;


  //move data from cpu to gpu
	cudaMemcpy(x_gpu, x_cpu, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y_cpu, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

	gettimeofday(&start, NULL);
  cpu_saxpy(ARRAY_SIZE,1.0f, x_cpu, y_cpu);
  gettimeofday(&end, NULL);
  printf("Computing SAXPY on the CPU… Done!\n");
  printf("CPU saxpy: %f milliseconds.\n", timeeval(start, end));
	gettimeofday(&start, NULL);
	gpu_saxpy<<<(ARRAY_SIZE + blocksize - 1)/blocksize, blocksize>>>(ARRAY_SIZE, 1.0f, x_gpu, y_gpu);
  gettimeofday(&end, NULL);
  printf("Computing SAXPY on the GPU… Done!\n");
  printf("GPU saxpy: %f milliseconds.\n", timeeval(start, end));

  //move data from gpu to cpu
	cudaMemcpy(y_cpu_res, y_gpu, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	if(cmp_saxpy(ARRAY_SIZE, y_cpu, y_cpu_res)==TRUE)
		printf("Succesful\n");
	else
		printf("Failed\n");

	cudaFree(x_gpu);
	cudaFree(y_gpu);
	return 0;
}
