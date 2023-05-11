#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda.h>
#include "compute.h"
#include <cuda_runtime.h>


#define BLOCKSIZE 256


// Device function, to compute the pairwise acceleration between objects
__device__ void computeAcceleration(const vector3* __restrict__ pos, const double* __restrict__ mass,  vector3* __restrict__ accels)
{
    // Compute the thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread ID is within the valid range of entities
    if (tid >= NUMENTITIES)
        return;

    // Compute the pairwise accelerations between objects
    for (int i = 0; i < NUMENTITIES; i++)
    {
        if (i == tid)
        {
            accels[tid][0] = 0.0;
            accels[tid][1] = 0.0;
            accels[tid][2] = 0.0;
        }
        else
        {
            vector3 distance;
            distance[0] = pos[tid][0] - pos[i][0];
            distance[1] = pos[tid][1] - pos[i][1];
            distance[2] = pos[tid][2] - pos[i][2];

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1.0 * GRAV_CONSTANT * mass[i] / magnitude_sq;

            accels[tid][0] += accelmag * distance[0] / magnitude;
            accels[tid][1] += accelmag * distance[1] / magnitude;
            accels[tid][2] += accelmag * distance[2] / magnitude;
        }
    }
}

// Kernel function
__global__ void compute(vector3* __restrict__ pos, vector3* __restrict__ vel, const double* __restrict__ mass)
{
    //Compute the thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if the thread ID is within the valid range of entities
    if (tid >= NUMENTITIES)
        return;

    // Declare an array to store the accelerations for each entity
    vector3 accels[NUMENTITIES];

    // Compute the acceleration for each entity
    computeAcceleration(pos, mass, accels);

    // Retrieve the acceleration values for the current entity
    double ax = accels[tid][0];
    double ay = accels[tid][1];
    double az = accels[tid][2];

    // Update the velocity and position of the current entity
    vel[tid][0] += ax * INTERVAL;
    vel[tid][1] += ay * INTERVAL;
    vel[tid][2] += az * INTERVAL;

    pos[tid][0] += vel[tid][0] * INTERVAL;
    pos[tid][1] += vel[tid][1] * INTERVAL;
    pos[tid][2] += vel[tid][2] * INTERVAL;
}
