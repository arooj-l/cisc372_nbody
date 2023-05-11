#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"
#include <cuda.h>

#define BLOCKSIZE 256

__device__ void computeAcceleration(const vector3* __restrict__ pos, const double* __restrict__ mass,  vector3* __restrict__ accels);

__global__ void compute(vector3* __restrict__ pos,  vector3* __restrict__ vel,const double* __restrict__ mass);
/*
// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass;
*/

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);

    // Allocate and initialize memory on the host
    vector3* hPos = new vector3[NUMENTITIES];
    vector3* hVel = new vector3[NUMENTITIES];
    double* hMass = new double[NUMENTITIES];
	initHostMemory(NUMENTITIES);

    // Copy data from host to device
    vector3* dPos;
    vector3* dVel;
    double* dMass;
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	#endif
	//allocate device memory for position and velocity arrays
    cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES);

    //copy initial positions and velocities, mass from host to device
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
    
	// Set up the execution configuration
    int numBlocks = (NUMENTITIES + BLOCKSIZE - 1) / BLOCKSIZE;
	
	//perform N-body simulation for the specified duration and interval
    for (t_now = 0; t_now < DURATION; t_now += INTERVAL)
    {
        compute<<<numBlocks, BLOCKSIZE>>>(d_hPos, d_hVel, d_mass);
        cudaDeviceSynchronize();
	}

    // Copy final positions and velocities from device to host
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

	// Clean up memory on the host
    delete[] hPos;
    delete[] hVel;
    delete[] hMass;
/*
   //clean up memory on the host
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
*/
	/*
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute();
	}
	*/
	clock_t t1=clock()-t0;
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();

	return 0; 

}
