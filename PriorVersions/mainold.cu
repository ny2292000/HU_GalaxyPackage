//============================================================================
// Name        : GalaxyFormation.cpp
// Author      : Marco Pereira
// Version     : 1.0.0
// Copyright   : Your copyright notice
// Description : Hypergeometrical Universe Galaxy Formation in C++, Ansi-style
//============================================================================
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include <vector>
#include <memory>  // for std::unique_ptr
#include <cmath>
#include <stdio.h>
#include <stdexcept>
#include <cstring>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include <array>
#include "/usr/include/boost/python.hpp"
#include <iostream>
#include <future>
#include <nlopt.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>

double pi = 3.141592653589793;

void print( const std::vector<double>  a) {
    std::cout << "The vector elements are : ";
    for (int i = 0; i < a.size(); i++)
        std::cout << a.at(i) << ' ';
}

std::vector<double> createCUDAVector(  const  std::vector<double>  hostPointer){
    std::stringstream ss;
    ss << hostPointer;
    std::string pointerStr = ss.str() + "_device";
    int nx= sizeof (hostPointer);
    std::vector<double> newPointer;
    cudaMalloc((void**)&newPointer, nx * sizeof(double));
    cudaMemcpy(newPointer, hostPointer.data(), hostPointer.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cout << "Error copying host pointer to device: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    cudaPointerAttributes attributes;
    error = cudaPointerGetAttributes(&attributes, hostPointer);
    if (error != cudaSuccess) {
        std::cout << "Error getting pointer attributes: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    return newPointer;
}

void deleteCUDAVector(double * x_vector_host, double * x_vector_device){
    delete[] x_vector_host;
    cudaFree(x_vector_device);
}

std::pair<dim3, dim3> get_block_size(int n, int threads_per_block) {
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    dim3 block_size(threads_per_block, 1, 1);
    dim3 num_blocks(blocks_per_grid, 1, 1);
    return std::make_pair(num_blocks, block_size);
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
    PyObject *array = PyArray_SimpleNew(1, &size, NPY_DOUBLE);

    // Copy the input vector data to the array data
    double *data_ptr = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(array)));
    memcpy(data_ptr, vec.data(), size * sizeof(double));

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


// Returns a vector of zeros with the given size
std::vector<double> zeros(int size) {
    return std::vector<double>(size, 0.0);
}




// CUDA kernel to compute the gravitational acceleration f_z
// for all points in r and z
__global__ void get_all_g_kernel(double G, const double* dv0, const double* r_sampling,
                                 const double* z_sampling, const double* r, const double* z, const double* costheta,
                                 const double* sintheta, const double* rho, int costheta_size, bool radial,
                                 double* f_z, int nr_sampling, int nz_sampling,  int nr, int nz, int irr, int izz) {
    // Get the indices of the point in r and z for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nr_sampling && j < nz_sampling) {
        // Initialize the result variable for this thread
        double res = 0.0;
        // Loop over all r and z points in the sampling vectors
        for (int ir = 0; ir < nr; ir++) {
//            if (radial && ( r[ir] > r_sampling[irr] )) {
//                break;
//            }
            for (int iz = 0; iz < nz; iz++) {
                // Loop over all theta angles in the costheta vector
                for (int k = 0; k < costheta_size; k++) {
                    // Compute the distance between the sampling point and the point in r and z for this thread
                    double d = pow(z_sampling[j] - z[iz], 2.0) + pow(r_sampling[i] - r[ir] * costheta[k], 2.0)
                               + pow(r[ir] * sintheta[k], 2.0);
                    // Compute the contribution to the result variable for this thread from this sampling point
                    if (radial) {
                        res += G * rho[ir * nz + iz] * r[ir] * dv0[ir * nz + iz] *
                               (r_sampling[i] - r[ir] * costheta[k]) / pow(d, 1.5);
                    } else {
                        res += G * rho[ir * nz + iz] * r[ir] * dv0[ir * nz + iz] *
                               (z_sampling[j] - z[iz]) / pow(d, 1.5);
                    }
                }
            }
        }
        // Store the result variable for this thread in the output array
        f_z[i + j * nr] = res;
    }
}

std::vector<double>
get_all_g_impl_cuda(double G, const std::vector<double>& dv0, const std::vector<double>& r_sampling,
                    const std::vector<double>& z_sampling, const std::vector<double>& r, const std::vector<double>& z,
                    const std::vector<double>& costheta, const std::vector<double>& sintheta,
                    const std::vector<double>& rho, bool radial) {

    int nr_sampling = r_sampling.size();
    int nz_sampling = z_sampling.size();
    int nr = r.size();
    int nz = z.size();
    int nrho = rho.size();
    int ndev0 = dv0.size();

    std::vector<double> dev_dv0 = createCUDAVector(dv0);
    std::vector<double> dev_r_sampling;
    std::vector<double> dev_z_sampling;
    std::vector<double> dev_r;
    std::vector<double> dev_z;
    std::vector<double> dev_costheta;
    std::vector<double> dev_sintheta;
    std::vector<double> dev_rho;
    std::vector<double> dev_f_z;

    cudaMalloc((void**)&dev_dv0,  ndev0 * sizeof(double));
    cudaMalloc((void**)&dev_r_sampling, nr_sampling * sizeof(double));
    cudaMalloc((void**)&dev_z_sampling, nz_sampling * sizeof(double));
    cudaMalloc((void**)&dev_r, nr * sizeof(double));
    cudaMalloc((void**)&dev_z, nz * sizeof(double));
    cudaMalloc((void**)&dev_costheta, costheta.size() * sizeof(double));
    cudaMalloc((void**)&dev_sintheta, sintheta.size() * sizeof(double));
    cudaMalloc((void**)&dev_rho, nrho * sizeof(double));
    cudaMalloc((void**)&dev_f_z, nr_sampling * nz_sampling * sizeof(double));


    cudaMemcpy(dev_dv0, dv0.data(), dv0.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r_sampling, r_sampling.data(), nr_sampling * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z_sampling, z_sampling.data(), nz_sampling * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, r.data(), nr * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, z.data(), nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_costheta, costheta.data(), costheta.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sintheta, sintheta.data(), sintheta.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rho, rho.data(), nrho * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 num_blocks((nr_sampling + block_size.x - 1) / block_size.x, (nz_sampling + block_size.y - 1) / block_size.y);

    for (int ir = 0; ir < nr_sampling; ir++) {
        for (int iz = 0; iz < nz_sampling; iz++) {
            // Launch kernel
            get_all_g_kernel<<<num_blocks, block_size>>>(G, dev_dv0, dev_r_sampling+ir, dev_z_sampling+iz,
                                                         dev_r, dev_z, dev_costheta, dev_sintheta, dev_rho,
                                                         costheta.size(), radial, dev_f_z,
                                                         nr_sampling, nz_sampling, nr, nz, ir, iz);
            cudaDeviceSynchronize();
        }
    }

    std::vector<double> f_z(nr_sampling * nz_sampling);
    cudaMemcpy(f_z.data(), dev_f_z, nr_sampling * nz_sampling * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dev_dv0);
    cudaFree(dev_r_sampling);
    cudaFree(dev_z_sampling);
    cudaFree(dev_r);
    cudaFree(dev_z);
    cudaFree(dev_costheta);
    cudaFree(dev_sintheta);
    cudaFree(dev_rho);
    cudaFree(dev_f_z);

    return f_z;
}

// # CPU functions

double
get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G, const std::vector<double> &dv0,
          const std::vector<double> &r,
          const std::vector<double> &z,
          const std::vector<double> &costheta, const std::vector<double> &sintheta, const std::vector<double> &rho,
          bool radial) {
    unsigned int nr = r.size();
    unsigned int nz = z.size();
    unsigned int ntheta = costheta.size();
    double res = 0.0;
    for (unsigned int i = 0; i < nr; i++) {
        if (radial && (r[i] > r_sampling_ii)) {
            break;
        }
        for (unsigned int j = 0; j < nz; j++) {
            for (unsigned int k = 0; k < ntheta; k++) {
                double d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
                           r[i] * r[i] * costheta[k] * costheta[k];
                if (radial) {
                    res += G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
                } else {
                    res += G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
                }

            }
        }
    }
    return res;
}
// ##################################################################


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
            futures.emplace_back(
                    std::async(get_g_cpu, r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho, radial));
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





std::vector<double>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<double> &rho, bool radial = true, bool cuda = true) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    // G = cc.G.to(uu.km / uu.s ** 2 / uu.kg * uu.lyr ** 2).value *(1+redshift)
    double G = 7.456866768350099e-46 * (1 + redshift);
    if (device_count > 0 && cuda) {
        // If CUDA devices are available, use the CUDA implementation
        return get_all_g_impl_cuda(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, radial);
    } else {
        // If no CUDA devices are available, use the CPU implementation
        return get_all_g_impl_cpu(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, radial);
    }
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
    bool cuda = true;
    std::vector<double> v_r(nr_sampling);
    std::vector<double> f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z,
                                        costheta, sintheta, rho, radial, cuda);
    // Calculate velocities
    double v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z[i] * r_sampling[i] * km_lyr;
        v_r[i] = sqrt(v_squared); // 9460730777119.56 km
    }
    // Return result
    return v_r;
}


std::vector<double> creategrid(double rho_0, double alpha_0, double rho_1, double alpha_1, int n) {
    if (alpha_1 > alpha_0) {
        double alpha_ = alpha_0;
        double rho_ = rho_0;
        alpha_0 = alpha_1;
        rho_0 = rho_1;
        alpha_1 = alpha_;
        rho_1 = rho_;
    }
    int n_range = 4;
    double r_max_1 = n_range / alpha_0;
    double r_max_2 = n_range / alpha_1;
    double M1 = massCalc(alpha_0, rho_0, 1.0);
    double M2 = massCalc(alpha_1, rho_1, 1.0);
    int n1 = M1 / (M1 + M2) * n;
    int n2 = M2 / (M1 + M2) * n;
    double r_min1 = 1.0;
    double r_min2 = r_max_1 + 1.0;

    // Define the grid of n points using a geometric sequence
    std::vector<double> r(n1 + n2);
    for (int i = 0; i < n1; i++) {
        r[i] = r_min1 * pow(r_max_1 / r_min1, i / (double) (n1 - 1));
    }
    for (int i = 0; i < n2; i++) {
        r[i + n1] = r_min2 * pow(r_max_2 / r_min2, i / (double) (n2 - 1));
    }
    return r;
}


class Galaxy {       // The class
public:             // Access specifier
    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    double R_max;
    double Mtotal_si;
    const double pi = 3.141592653589793;
    double alpha_0;
    double rho_0;
    double alpha_1;
    double rho_1;
    double h0;
    double dz;
    double redshift;
    double GalaxyMass;
    std::vector<double> r;
    std::vector<double> dv0;
    std::vector<double> z;
    std::vector<double> r_sampling;
    std::vector<double> z_sampling;
    std::vector<double> rho;
    std::vector<double> theta;
    std::vector<double> costheta;
    std::vector<double> sintheta;
    std::vector<double> f_z;
    std::vector<double> rotational_velocity_;
    // ######################################
    std::vector<double> x_rotation_points;
    int n_rotation_points = 0;
    std::vector<double> v_rotation_points;
    // ######################################
    bool radial = true;


    Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0)
            : R_max(R_max), nr(nr), nz(nz), nr_sampling(nr_sampling), nz_sampling(nz_sampling),
              alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), redshift(redshift),
              GalaxyMass(GalaxyMass) {
        // ######################################
        r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
        nr = r.size();
        z = linspace(-h0 / 2.0, h0 / 2.0, nz);
        rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        theta = linspace(0, 2 * pi, ntheta);
        costheta = costhetaFunc(theta);
        sintheta = sinthetaFunc(theta);
        // Allocate result vector
        z_sampling = linspace(-h0 / 2.0, h0 / 2.0, nz);
        r_sampling = linspace(1, R_max, nr_sampling);
        dz = h0 / nz;
        double dtheta = 2 * pi / ntheta;
        dv0.resize(1);
        dv0[0] = 0.0;
        for (int i = 1; i < nr; i++) {
            dv0.push_back((r[i] - r[i - 1]) * dz * dtheta);
        }
    }


    std::vector<double> get_f_z(const std::vector<double> &x, bool radial = true, bool cuda = true) {
        // Calculate the rotation velocity using the current values of x
        double rho_0 = x[0];
        double alpha_0 = x[1];
        double rho_1 = x[2];
        double alpha_1 = x[3];
        double h0 = x[4];
        // Calculate the total mass of the galaxy
        std::vector<double> r_sampling = this->x_rotation_points;
        std::vector<double> z_sampling = {0.0};
        std::vector<double> f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z,
                                            costheta, sintheta, rho, radial, cuda);
        return f_z;
    }


    // Define the function to be minimized
    double error_function(const std::vector<double> &x) {
        // Calculate the rotation velocity using the current values of x
        double rho_0 = x[0];
        double alpha_0 = x[1];
        double rho_1 = x[2];
        double alpha_1 = x[3];
        double h0 = x[4];
        // Calculate the total mass of the galaxy
        double Mtotal_si = massCalc(alpha_0, rho_0, h0);  // Mtotal in Solar Masses
        double error_mass = pow((this->GalaxyMass - Mtotal_si) / this->GalaxyMass, 2);
        std::vector<double> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        std::vector<double> vsim = calculate_rotational_velocity(this->redshift, this->dv0,
                                                                 this->x_rotation_points,
                                                                 this->r,
                                                                 this->z,
                                                                 this->costheta,
                                                                 this->sintheta,
                                                                 rho);
        double error = 0.0;
        for (int i = 0; i < n_rotation_points; i++) { error += pow((v_rotation_points[i] - vsim[i]), 2); }
        std::cout << "Total Error = " << (error + error_mass) << "\n";
        return error + error_mass;
    }


    void read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin) {
        n_rotation_points = vin.size();
        this->x_rotation_points.clear();
        this->v_rotation_points.clear();
        for (const auto &row: vin) {
            this->x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
            this->v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
        }
    }

//// Objective function for optimization
    static double objective_wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        return reinterpret_cast<Galaxy *>(data)->error_function(x);
    }

    // Define the Nelder-Mead optimizer
    std::vector<double>
    nelder_mead(const std::vector<double> &x0, int max_iter = 1000, double xtol_rel = 1e-6) {
        nlopt::opt opt(nlopt::LN_NELDERMEAD, x0.size());
        opt.set_min_objective(&Galaxy::objective_wrapper, this);
        opt.set_xtol_rel(xtol_rel);
        std::vector<double> x = x0;
        double minf;
        nlopt::result result = opt.optimize(x, minf);
        if (result < 0) {
            std::cerr << "nlopt failed: " << strerror(result) << std::endl;
        }
        return x;
    }


    // End of Galaxy Class
};


int main() {
//    std::vector<double> arr = {1.0, 2.0, 3.0, 4.0, 5.0};
//    int n = arr.size();
//    double* dev_arr;
//    cudaMalloc((void**)&dev_arr, n * sizeof(double));
//    cudaMemcpy(dev_arr, arr.data(), n * sizeof(double), cudaMemcpyHostToDevice);
//
//    // Allocate memory for the result array
//    double* result = new double[n];
//    double* dev_result;
//    cudaMalloc((void**)&dev_result, n * sizeof(double));
//
//    // Copy the data back from the device to the host
//    cudaMemcpy(result, dev_arr, n * sizeof(double), cudaMemcpyDeviceToHost);
//
//    // Print the result to verify it was copied correctly
//    std::cout << "Result: ";
//    for (int i = 0; i < n; i++) {
//        std::cout << result[i] << " ";
//    }
//    std::cout << std::endl;
//
//    // Free the device memory
//    cudaFree(dev_arr);
//    cudaFree(dev_result);
//
//    // Free the host memory
//    delete[] result;


    std::vector<std::array<double, 2>> m33_rotational_curve = {
//            {0.0f,       0.0f},
//            {1508.7187f, 38.674137f},
//            {2873.3889f, 55.65067f},
//            {4116.755f,  67.91063f},
//            {5451.099f,  79.22689f},
//            {6846.0957f, 85.01734f},
//            {8089.462f,  88.38242f},
//            {9393.48f,   92.42116f},
//            {10727.824f, 95.11208f},
//            {11880.212f, 98.342697f},
//            {13275.208f, 99.82048f},
//            {14609.553f, 102.10709f},
//            {18521.607f, 104.25024f},
//            {22403.336f, 107.60643f},
//            {26406.369f, 115.40966f},
//            {30379.076f, 116.87875f},
//            {34382.107f, 116.05664f},
//            {38354.813f, 117.93005f},
            {42266.87f,  121.42091f},
//            {46300.227f, 128.55017f},
//            {50212.285f, 132.84966f}
    };


    const double M33_Distance = 3.2E6;
    const double Radius_Universe_4D = 14.03E9;
    double redshift = M33_Distance / (Radius_Universe_4D - M33_Distance);
    const int nr = 100;
    const int nz = 101;
    const int ntheta = 102;
    const int nr_sampling = 103;
    const int nz_sampling = 104;
    const double R_max = 50000.0;
    const double pi = 3.141592653589793;
    const double alpha_0 = 0.00042423668409927005;
    const double rho_0 = 12.868348904393013;
    const double alpha_1 = 2.0523892233327836e-05;
    const double rho_1 = 0.13249804158174094;
    const double h0 = 156161.88949004377;
    const double GalaxyMass = 5E10;
//    const std::vector<double> r = linspace(1,R_max, nr);
    std::vector<double> r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    const std::vector<double> z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    const std::vector<double> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
    const std::vector<double> theta = linspace(0, 2 * pi, ntheta);
    std::vector<double> f_z = zeros(nr * nz);
    Galaxy M33 = Galaxy(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
                        R_max, nr, nz, nr_sampling, nz_sampling, ntheta,
                        redshift);
    M33.read_galaxy_rotation_curve(m33_rotational_curve);
    std::vector<double> x0 = {rho_0, alpha_0, rho_1, alpha_1, h0};
//    std::vector<double> xout = {rho_0, alpha_0, rho_1, alpha_1, h0};
//    std::vector<double> xout = {22.0752, 0.00049759, 0.122031, 1.71929e-05, 125235};
//    xout = M33.nelder_mead(x0);
//    print(xout);

    bool radial = true;
    bool cuda = false;
//    print(M33.get_f_z(x0, radial, cuda));
    cuda = true;
    print(M33.get_f_z(x0, radial, cuda));

}