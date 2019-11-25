#include <stdio.h>
#include <sys/time.h>
#define TPB 16
#define NUM_PARTICLES 1000000
#define NUM_ITTERATIONS 5

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
	float const round_err = 0.0001;
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
	const float err = 0.01;
	for(int i = 0; i < n; i++)
	{
		if(fabs(cpu_p[i].position.x - gpu_p[i].position.x) >= err) 
			return FALSE;
		else if (fabs(cpu_p[i].position.y - gpu_p[i].position.y) >= err)
			return FALSE;
		else if (fabs(cpu_p[i].position.z - gpu_p[i].position.z) >= err)
                        return FALSE;
		else if (fabs(cpu_p[i].velocity.x - gpu_p[i].velocity.x) >= err)
                        return FALSE;
		else if (fabs(cpu_p[i].velocity.y - gpu_p[i].velocity.y) >= err)
                        return FALSE;
		else if (fabs(cpu_p[i].velocity.z - gpu_p[i].velocity.z) >= err)
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
	
	Particle *particles;
	
	// initinalization of particles
        cudaMallocManaged(&particles, NUM_PARTICLES*sizeof(Particle));
	int i;
	for(i = 0; i < NUM_PARTICLES; i++)
	{
		particles[i].position = gen_random_float3();
		particles[i].velocity = gen_random_float3();
	}

	//Particle *gpu_particles;
	Particle *result_particles = (Particle*)malloc(NUM_PARTICLES*sizeof(Particle));
	result_particles = particles;

	for(i = 0; i < NUM_ITTERATIONS; i++)
	{
		update<<<(NUM_PARTICLES + TPB - 1)/TPB, TPB>>>(particles, 1);
	}
	cudaDeviceSynchronize();	
	for(i = 0; i < NUM_ITTERATIONS; i++)
	{
		cpu_update(NUM_PARTICLES, result_particles, 1);
	}	
		
	if(cmp_particles(NUM_PARTICLES, particles, result_particles))
		printf("Succesful\n");
	else
		printf("Failed\n");
	
	cudaFree(particles);

	return 0;
}
