#include <cmath>
#include <vector>
#include <future>
#include <iostream>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
//#include <cuda_runtime.h>
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/sort.h>









// The function get_all_g_cuda takes in the gravitational constant G, the volume element dv0, the sampling
// vectors r_sampling and z_sampling, and the integration variables r, z, and costheta. It also takes in
// the density vector rho and an optional redshift parameter, which is used to correct the gravitational
// constant G for the given epoch.
//
// The kernel function get_all_g_kernel performs the computation of the gravitational acceleration by
// looping over the sampling vectors and integration variables. It uses the density rho and the integration
// variables to calculate the gravitational acceleration at each point, and sums up the contributions from
// all elements of volume dv0 using a nested loop. The result is stored in the output vector f_z.
//
// The computation is performed on the GPU using CUDA, and the memory is allocated and copied using RAII and
// smart pointers to ensure proper memory management. The computation is parallelized using CUDA blocks and
// threads, with the number of blocks and threads determined based on the input size.








//__global__ void get_all_g_kernel(double G, const double* dv0, const double* r_sampling, const double* z_sampling,
//                                 const double* r, const double* z, const double* costheta, const double* sintheta,
//                                 const double* rho, int costheta_size, bool radial, double* f_z, int nr_sampling,
//                                 int nz_sampling, int nr, int nz, int ir, int iz)
//{
//    // Compute the global index for this thread
//    int aa = 1;
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (i < nr_sampling && j < nz_sampling) {
//        // Compute the distance between the sampling point and the point in r and z for this thread
//        double d = (z_sampling[j] - z[iz], 2.0) + (r_sampling[i] - r[ir] * costheta[0], 2.0)
//                   + (r[ir] * sintheta[0], 2.0);
//
//        // Compute the contribution to the result variable for this thread from this sampling point
//        double res = 0.0;
//        if (radial) {
//            res += G * rho[ir * nz + iz] * r[ir] * dv0[ir * nz + iz] *
//                   (r_sampling[0] - r[ir] * costheta[0]) / (d, 1.5);
//        } else {
//            res += G * rho[ir * nz + iz] * r[ir] * dv0[ir * nz + iz] *
//                   (z_sampling[0] - z[iz]) / (d, 1.5);
//        }
//
//        // Store the result variable for this thread in the output array
//        f_z[i + j * nr_sampling] = i+j;
//    }
//}





