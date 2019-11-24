#include <stdio.h>
#include <sys/time.h>
#define TPB 16
#define NUM_PARTICLES 1000000
#define NUM_ITTERATIONS 10

#define TRUE 1
#define FALSE 0

struct Particle
{
	float3 position;
	float3 velocity;
};

__device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(const float3 &a, const int &b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ int operator!=(const float3 &a, const float3 &b)
{
	const float round_err = 0.0001;
	if(fabs(a.x - b.x) >= round_err)
		return TRUE;
	else if(fabs(a.y - b.y) >= round_err)
		return TRUE;
	else if(fabs(a.z - b.z) >= round_err)
		return TRUE;
	else
		return FALSE;
}

float3 gen_random_float3()
{
	static float MAX_GEN = 10.0;
	return make_float3((float)rand()/(float)(RAND_MAX/MAX_GEN), (float)rand()/(float)(RAND_MAX/MAX_GEN), (float)rand()/(float)(RAND_MAX/MAX_GEN));
}

__global__ void update(Particle* a, int dt)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float3 delta_velocity = make_float3(1,2,3); //gen_random_float3();
	a[i].velocity = a[i].velocity +  delta_velocity;
	a[i].position = a[i].position +  a[i].velocity * dt;

}

void cpu_update(int n, Particle* a, int dt)
{
	for(int i = 0; i < n; i++)
	{
		a[i].velocity.x += 1;
	  a[i].velocity.y += 2;
		a[i].velocity.z += 3;
		a[i].position.x += a[i].velocity.x * dt;
		a[i].position.y += a[i].velocity.y * dt;
		a[i].position.z += a[i].velocity.z * dt;
	}
}

int cmp_particles(int n, Particle* cpu_p, Particle* gpu_p)
{
	for(int i = 0; i < n; i++)
	{
		if(cpu_p[i].position != gpu_p[i].position || cpu_p[i].velocity != gpu_p[i].velocity)
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

	//Particle particles[NUM_PARTICLES];
	Particle *particles = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));

	for(int i = 0; i < NUM_PARTICLES; i++)
	{
		particles[i].position = gen_random_float3();
		particles[i].velocity = gen_random_float3();
	}

	Particle *gpu_particles;
	Particle *results_particles = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
	cudaMalloc(&gpu_particles, NUM_PARTICLES*sizeof(Particle));
  struct timeval start, end;
  gettimeofday(&start, NULL);
	cudaMemcpy(gpu_particles, particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyHostToDevice);
  for(int i = 0; i < NUM_ITTERATIONS; i++){
		update<<<(NUM_PARTICLES + TPB - 1)/TPB, TPB>>>(gpu_particles, 1);
	}
  gettimeofday(&end, NULL);
  printf("GPU execution time: %f milliseconds.\n", timeeval(start, end));

	gettimeofday(&start, NULL);
	for(int i = 0; i < NUM_ITTERATIONS; i++)
	{
		cpu_update(NUM_PARTICLES, particles, 1);
	}
	gettimeofday(&end, NULL);
	printf("CPU execution time: %f milliseconds.\n", timeeval(start, end));

	cudaMemcpy(results_particles, gpu_particles, NUM_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);

	if(cmp_particles(NUM_PARTICLES, particles, results_particles))
		printf("Succesful\n");
	else
		printf("Failed\n");

	cudaFree(gpu_particles);

	return 0;
}
