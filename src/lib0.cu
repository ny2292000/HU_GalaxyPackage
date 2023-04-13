#include <cmath>
#include <vector>
#include <future>
#include <iostream>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

double pi = 3.141592653589793;

// Returns a vector of zeros with the given size
std::vector<double> zeros(int size) {
    return std::vector<double>(size, 0.0);
}

// # CPU functions

double
get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G, const std::vector<double> &dv0, const std::vector<double> &r,
          const std::vector<double> &z,
          const std::vector<double> &costheta, const std::vector<double> &sintheta, const std::vector<double> &rho, bool radial) {
    unsigned int nr = r.size();
    unsigned int nz = z.size();
    unsigned int ntheta = costheta.size();
    double res = 0.0;
    for (unsigned int i = 0; i < nr; i++) {
        if (r[i]> r_sampling_ii){
            break;}
        for (unsigned int j = 0; j < nz; j++) {
            for (unsigned int k = 0; k < ntheta; k++) {
                double d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
                           r[i] * r[i] * costheta[k] * costheta[k];
                if(radial){
                    res += G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
                }else{
                    res += G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
                }

            }
        }
    }
    return res;
}
// ##################################################################




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




// # CPU functions

std::vector<double>
get_all_g_impl_cpu(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
                   const std::vector<double> &z_sampling,
                   const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
                   const std::vector<double> &sintheta, const std::vector<double> &rho, bool radial = true) {
    std::vector<std::future<double>> futures;
    int nr = r_sampling.size();
    int nz = z_sampling.size();
    futures.reserve(nr * nz);
    // Spawn threads
    std::vector<double> f_z = zeros(nr * nz);
    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            futures.emplace_back(std::async(get_g_cpu, r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho, radial));
        }
    }

// Collect results and populate f_z
    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            f_z[i + j * nr] = futures[i * nz + j].get();
        }
    }
    return f_z;
}


// CUDA kernel to compute the gravitational acceleration f_z
// for all points in r and z
__global__ void get_all_g_kernel(int nr, int nz, int nr_sampling, int nz_sampling, double G, double *dv0,
                                 const double *r_sampling, const double *z_sampling,
                                 const double *grid_data, const double *rho,
                                 int costheta_size, bool radial, double *f_z) {

    // Get the indices of the point in r and z for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Use shared memory for caching intermediate results
    extern __shared__ double shared_data[];

    if (i < nr && j < nz) {
        // Initialize the result variable for this thread
        double res = 0.0;
        // Loop over all r and z points in the sampling vectors
        for (int ir = 0; ir < nr_sampling; ir++) {
            for (int iz = 0; iz < nz_sampling; iz++) {
                // Loop over all theta angles in the costheta vector
                for (int k = 0; k < costheta_size; k++) {
                    // Compute the distance between the sampling point and the point in r and z for this thread
                    double d = pow(z_sampling[iz] - grid_data[nz + j], 2.0) +
                               pow(r_sampling[ir] - grid_data[i] * grid_data[2 * nz + k], 2.0)
                               + grid_data[i] * grid_data[i] * grid_data[3 * nz + k] * grid_data[3 * nz + k];
                    // Compute the contribution to the result variable for this thread from this sampling point
                    if (radial) {
                        res += G * rho[ir * nz + iz] * grid_data[i] * dv0[i * nz + j] *
                               (r_sampling[ir] - grid_data[i] * grid_data[2 * nz + k]) / pow(d, 1.5);
                    } else {
                        res += G * rho[ir * nz + iz] * grid_data[i] * dv0[i * nz + j] *
                               (z_sampling[iz] - grid_data[nz + j]) / pow(d, 1.5);
                    }
                }
            }
        }
        // Store the result variable for this thread in the output array
        f_z[i + j * nr] = res;
    }
}


std::vector<double>
get_all_g_impl_cuda(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
                    const std::vector<double> &z_sampling,
                    const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
                    const std::vector<double> &sintheta, const std::vector<double> &rho, bool radial = true) {
//    This function computes the gravitational force at all points in a 2D grid using CUDA. The function takes
//    in several input parameters:
//
//            G: the gravitational constant.
//            dv0: the volume element.
//            r_sampling and z_sampling: the radial and vertical sampling vectors, respectively.
//            r, z, and costheta, sintheta, rho: the integration variables used to add up the contribution
//            from elements of volume dv0.
//            redshift: the epoch value.
//            radial: a boolean indicating whether to compute the gravitational force radially or spherically.
//
//            The function first allocates and copies device memory using RAII and smart pointers. It then
//            launches a kernel to compute the gravitational force at all points in the grid. The kernel uses
//            the integration variables to add up the contribution from elements of volume dv0 and the sampling
//            vectors to calculate the distance between points. The kernel corrects G for epoch and computes the
//            gravitational force at all points in the grid. Finally, the function copies the results back to the
//            host and returns them as a vector of doubles.
    int nr_sampling = r_sampling.size();
    int nz_sampling = z_sampling.size();
    int nr = r.size();
    int nz = z.size();

    // Combine r, z, costheta, and sintheta into a single vector (grid_data) for cudaMemcpy
    std::vector<double> grid_data(r);
    grid_data.insert(grid_data.end(), z.begin(), z.end());
    grid_data.insert(grid_data.end(), costheta.begin(), costheta.end());
    grid_data.insert(grid_data.end(), sintheta.begin(), sintheta.end());

    // Allocate and copy device memory using thrust::device_vector
    thrust::device_vector<double> dev_r_sampling(r_sampling.begin(), r_sampling.end());
    thrust::device_vector<double> dev_z_sampling(z_sampling.begin(), z_sampling.end());
    thrust::device_vector<double> dev_grid_data(grid_data.begin(), grid_data.end());
    thrust::device_vector<double> dev_rho(rho.begin(), rho.end());
    thrust::device_vector<double> dev_dv0(dv0.begin(), dv0.end());
    thrust::device_vector<double> dev_f_z(r.size() * z.size());

    // Launch kernel
    int costheta_size = costheta.size();
    dim3 block_size(32, 32);
    dim3 num_blocks((r.size() + block_size.x - 1) / block_size.x, (z.size() + block_size.y - 1) / block_size.y);

    // G already corrected for epoch
    get_all_g_kernel<<<num_blocks, block_size>>>(r.size(), z.size(), r_sampling.size(), z_sampling.size(), G,
            thrust::raw_pointer_cast(dev_dv0.data()),
            thrust::raw_pointer_cast(dev_r_sampling.data()),
            thrust::raw_pointer_cast(dev_z_sampling.data()),
            thrust::raw_pointer_cast(dev_grid_data.data()),
            thrust::raw_pointer_cast(dev_rho.data()), costheta_size,
            radial, thrust::raw_pointer_cast(dev_f_z.data()));

    // Copy results back to host using thrust::copy
    std::vector<double> f_z(dev_f_z.size());
    thrust::copy(dev_f_z.begin(), dev_f_z.end(), f_z.begin());

    // Return result
    return f_z;
}


std::vector<double>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<double> &rho, bool radial = true) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    device_count = 0;
    // G = cc.G.to(uu.km / uu.s ** 2 / uu.kg * uu.lyr ** 2).value *(1+redshift)
    double G = 7.456866768350099e-46 * (1 + redshift);
    if (device_count > 0) {
        // If CUDA devices are available, use the CUDA implementation
        return get_all_g_impl_cuda(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, radial);
    } else {
        // If no CUDA devices are available, use the CPU implementation
        return get_all_g_impl_cpu(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, radial);
    }
}


double massCalcX(double alpha, double rho, double h, double x) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double M_si = -2 * pi * h * rho * x * exp(-alpha * x) / alpha - 2 * pi * h * rho * exp(-alpha * x) / pow(alpha, 2) +
                  2 * pi * h * rho / pow(alpha, 2);
    M_si = M_si * factor;
    return M_si;
}


double massCalc(double alpha, double rho, double h) {
    double factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    double Mtotal_si = 2 * pi * h * rho / pow(alpha, 2);
    Mtotal_si = Mtotal_si * factor;
    return Mtotal_si;
}


std::vector<double> calculate_rotational_velocity(double redshift, const std::vector<double> &dv0,
                                                  std::vector<double> r_sampling,
                                                  const std::vector<double> &r,
                                                  const std::vector<double> &z,
                                                  const std::vector<double> &costheta,
                                                  const std::vector<double> &sintheta,
                                                  const std::vector<double> &rho) {
//#######################################################################################################
//#######################################################################################################
//#######################################################################################################
//#######################################################################################################
//This function takes in two vectors: f_z, which contains the gravitational acceleration values calculated by
// the get_all_g_cuda() function, and r_sampling, which contains the radial sampling positions used in the calculation.
//
//The function first calculates the dimensions of the f_z array based on its size. Then, it allocates a result
// vector v_r with the same size as r_sampling.
//
//Next, the function calculates the velocities for each radial position. It uses the formula v**2 = f_z / r_sampling
// to calculate the squared velocity at each radial position, then takes the square root to get the actual velocity.
// This calculation is done in a loop over each position in r_sampling.
//
//Finally, the function returns the v_r vector containing the calculated velocities.
    int nr_sampling = r_sampling.size();
    double km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    // Allocate result vector
    std::vector<double> z_sampling = {0.0};
    bool radial = true;
    std::vector<double> v_r(nr_sampling);
    std::vector<double> f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z,
                                        costheta, sintheta, rho, radial);
    // Calculate velocities
    double v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z[i] * r_sampling[i] * km_lyr;
        v_r[i] = sqrt(v_squared); // 9460730777119.56 km
    }
    // Return result
    return v_r;
}


std::vector<double> vec_from_array(PyArrayObject *array) {
    // Check that input is a 1-dimensional array of doubles
    if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_DOUBLE) {
        throw std::invalid_argument("Input must be a 1D NumPy array of doubles");
    }

    // Get the size of the array and a pointer to its data
    int size = PyArray_SIZE(array);
    double *data_ptr = static_cast<double *>(PyArray_DATA(array));

    // Create a vector from the array data
    std::vector<double> vec(data_ptr, data_ptr + size);

    return vec;
}


PyArrayObject *array_from_vec(std::vector<double> vec) {
    // Create a 1D NumPy array of the same size as the input vector
    npy_intp size = vec.size();
    PyObject * array = PyArray_SimpleNew(1, &size, NPY_DOUBLE);

    // Copy the input vector data to the array data
    double *data_ptr = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(array)));
    std::memcpy(data_ptr, vec.data(), size * sizeof(double));

    return reinterpret_cast<PyArrayObject *>(array);
}

std::vector<double> costhetaFunc(const std::vector<double> &theta) {
    unsigned int points = theta.size();
    std::vector<double> res(points);
    for (unsigned int i = 0; i < points; i++) {
        res[i] = cos(theta[i]);
    }
    return res;
}

std::vector<double> sinthetaFunc(const std::vector<double> &theta) {
    unsigned int points = theta.size();
    std::vector<double> res(points);
    for (unsigned int i = 0; i < points; i++) {
        res[i] = sin(theta[i]);
    }
    return res;
}


std::vector<double> linspace(double start, double end, size_t points) {
    std::vector<double> res(points);
    double step = (end - start) / (points - 1);
    size_t i = 0;
    for (auto &e: res) {
        e = start + step * i++;
    }
    return res;
}

std::vector<double> density(double rho_0, double alpha_0, double rho_1, double alpha_1, std::vector<double> r) {
    unsigned int vecsize = r.size();
    std::vector<double> density_(vecsize);
    // to kg/lyr^3
    rho_0 *= 1.4171253E27;  //(h_mass/uu.cm**3).to(uu.kg/uu.lyr**3) =<Quantity 1.41712531e+27 kg / lyr3>
    rho_1 *= 1.4171253E27;
    for (unsigned int i = 0; i < vecsize; i++) {
        density_[i] = rho_0 * exp(-alpha_0 * r[i]) + rho_1 * exp(-alpha_1 * r[i]);
    }
    return density_;
}
