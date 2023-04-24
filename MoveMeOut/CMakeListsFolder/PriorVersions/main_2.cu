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
#include <thread>
#include <memory>  // for std::unique_ptr
#include <cmath>
#include <stdexcept>
#include <cstring>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include <array>
#include "/usr/include/boost/python.hpp"
#include <iostream>
#include <future>

float pi =  3.141592653589793;


std::vector<float> zeros(int points)
{
    std::vector<float> res(points);
    for (unsigned int i=0; i<points; i++)
    {
        res[i] = 0.0;
    }
    return res;
}


float get_g_cpu( int ii, int jj, float G, float dv0, const std::vector<float>& r, const std::vector<float>& z,
                 const std::vector<float>& costheta, const std::vector<float>& sintheta, const std::vector<float>& rho)
{
    unsigned int nr = r.size();
    unsigned int nz = z.size();
    unsigned int ntheta = costheta.size();
    float res = 0.0;
    for(unsigned int i=0;i<nr;i++)
    {
        for(unsigned int j=0;j<nz;j++)
        {
            for(unsigned int k=0;k<ntheta;k++)
            {
                float d = pow(z[j] - z[jj], 2.0) + pow(r[ii] - r[i]*sintheta[k], 2.0) + r[i]*r[i]*costheta[k]*costheta[k];
                res += G*rho[i]*r[i]*dv0*(z[j] - z[jj])/pow( d, 1.5);
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
std::vector<float> get_all_g_impl_cpu(int nr, int nz, float G, float dv0, const std::vector<float>& r, const std::vector<float>& z,
                                      const std::vector<float>& costheta, const std::vector<float>& sintheta, const std::vector<float>& rho)
{
    std::vector<std::future<float>> futures;
    futures.reserve(nr * nz);
    // Spawn threads
    std::vector<float> f_z = zeros(nr*nz);
    for(unsigned int i = 0; i < nr; i++)
    {
        for(unsigned int j = 0; j < nz; j++)
        {
            futures.emplace_back(std::async(get_g_cpu, i, j, G, dv0, r, z, costheta, sintheta, rho));
        }
    }

// Collect results and populate f_z
    for(unsigned int i = 0; i < nr; i++)
    {
        for(unsigned int j = 0; j < nz; j++)
        {
            f_z[i + j * nr] = futures[i * nz + j].get();
        }
    }
    return f_z;
}



// CUDA kernel to compute the gravitational acceleration f_z
// for all points in r and z
__global__ void get_all_g_kernel(int nr, int nz, int nr_sampling, int nz_sampling, float G, float dv0,
                                 const float *r_sampling, const float *z_sampling,
                                 const float *grid_data, const float *rho,
                                 int costheta_size, bool radial, float *f_z) {

    // Get the indices of the point in r and z for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Use shared memory for caching intermediate results
    extern __shared__ float shared_data[];

    if (i < nr && j < nz) {
        // Initialize the result variable for this thread
        float res = 0.0;
        // Loop over all r and z points in the sampling vectors
        for (int ir = 0; ir < nr_sampling; ir++) {
            for (int iz = 0; iz < nz_sampling; iz++) {
                // Loop over all theta angles in the costheta vector
                for (int k = 0; k < costheta_size; k++) {
                    // Compute the distance between the sampling point and the point in r and z for this thread
                    float d = pow(z_sampling[iz] - grid_data[nz + j], 2.0) + pow(r_sampling[i] - grid_data[i] * grid_data[2 * nz + k], 2.0)
                              + grid_data[i] * grid_data[i] * grid_data[3 * nz + k] * grid_data[3 * nz + k];
                    // Compute the contribution to the result variable for this thread from this sampling point
                    if (radial) {
                        res += G * rho[ir] * grid_data[i] * dv0 * (r_sampling[ir] - grid_data[i] * grid_data[2 * nz + k]) / pow(d, 1.5);
                    } else {
                        res += G * rho[ir] * grid_data[i] * dv0 * (z_sampling[iz] - grid_data[nz + j]) / pow(d, 1.5);
                    }
                }
            }
        }
        // Store the result variable for this thread in the output array
        f_z[i + j * nr] = res;
    }
}


std::vector<float> get_all_g_impl_cuda(float G, float dv0,
                                       const std::vector<float> &r_sampling, const std::vector<float> &z_sampling,
                                       const std::vector<float> &r, const std::vector<float> &z,
                                       const std::vector<float> &costheta, const std::vector<float> &sintheta,
                                       const std::vector<float> &rho, float redshift = 0.0, bool radial = true) {
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
//            host and returns them as a vector of floats.
    int nr_sampling = r_sampling.size();
    int nz_sampling = z_sampling.size();
    int nr = r.size();
    int nz = z.size();

    // Combine r, z, costheta, and sintheta into a single vector (grid_data) for cudaMemcpy
    std::vector<float> grid_data(r);
    grid_data.insert(grid_data.end(), z.begin(), z.end());
    grid_data.insert(grid_data.end(), costheta.begin(), costheta.end());
    grid_data.insert(grid_data.end(), sintheta.begin(), sintheta.end());

    // Allocate and copy device memory
    float *dev_r_sampling, *dev_z_sampling, *dev_grid_data, *dev_rho, *dev_f_z;
    cudaMalloc((void **)&dev_r_sampling, nr_sampling * sizeof(float));
    cudaMemcpy(dev_r_sampling, r_sampling.data(), nr_sampling * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_z_sampling, nz_sampling * sizeof(float));
    cudaMemcpy(dev_z_sampling, z_sampling.data(), nz_sampling * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_grid_data, grid_data.size() * sizeof(float));
    cudaMemcpy(dev_grid_data, grid_data.data(), grid_data.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_rho, nr * nz * sizeof(float));
    cudaMemcpy(dev_rho, rho.data(), nr * nz * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_f_z, nr * nz * sizeof(float));

    // Launch kernel
    int costheta_size = costheta.size();
    dim3 block_size(32, 32);
    dim3 num_blocks((nr + block_size.x - 1) / block_size.x, (nz + block_size.y - 1) / block_size.y);

    // Correct G for epoch
    float G_corrected = G * (1 + redshift);
    get_all_g_kernel<<<num_blocks, block_size>>>(nr, nz, nr_sampling, nz_sampling, G_corrected, dv0,
                                                 dev_r_sampling, dev_z_sampling, dev_grid_data, dev_rho, costheta_size,
                                                 radial, dev_f_z);

    // Copy results back to host
    std::vector<float> f_z(nr * nz);
    cudaMemcpy(f_z.data(), dev_f_z, nr * nz * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_r_sampling);
    cudaFree(dev_z_sampling);
    cudaFree(dev_grid_data);
    cudaFree(dev_rho);
    cudaFree(dev_f_z);

    // Return result
    return f_z;
}



std::vector<float> get_all_g(float G, float dv0,
                             const std::vector<float> &r_sampling, const std::vector<float> &z_sampling,
                             const std::vector<float> &r, const std::vector<float> &z,
                             const std::vector<float> &costheta, const std::vector<float> &sintheta,
                             const std::vector<float> &rho, float redshift = 0.0, bool radial = true) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count > 0) {
        // If CUDA devices are available, use the CUDA implementation
        return get_all_g_impl_cuda(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, redshift, radial);
    } else {
        // If no CUDA devices are available, use the CPU implementation
        int nr = r.size();
        int nz = z.size();
        return get_all_g_impl_cpu(nr, nz, G, dv0, r, z, costheta, sintheta, rho);
    }
}









std::tuple<float> massCalc(float x, float alpha, float rho, float h) {
    float M_si = -2 * pi * h * rho * x * exp(-alpha * x) / alpha - 2 * pi * h * rho * exp(-alpha * x) / pow(alpha, 2) + 2 * pi * h * rho / pow(alpha, 2);
    M_si = M_si * 1.4171253E27;
    float Mtotal_si = 2 * pi * h * rho / pow(alpha, 2);
    Mtotal_si = Mtotal_si * 1.4171253E27;
    return  Mtotal_si;
}



std::vector<float> calculate_rotational_velocity(float G, float dv0,
                                                 std::vector<float> r_sampling,
                                                 const std::vector<float> &r,
                                                 const std::vector<float> &z,
                                                 const std::vector<float> &costheta,
                                                 const std::vector<float> &sintheta,
                                                 const std::vector<float> &rho,
                                                 float redshift = 0.0) {
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
    int nr = sqrt(r.size());

    // Allocate result vector
    std::vector<float> z_sampling;
    z_sampling.push_back(0.0);
    bool radial = true;
    std::vector<float> v_r(nr_sampling);
    std::vector<float> f_z = get_all_g(G, dv0, r_sampling, z_sampling, r, z,
                                            costheta, sintheta, rho, redshift, radial);
    // Calculate velocities
    for (int i = 0; i < nr_sampling; i++) {
        int idx = i * nr;
        float v_squared = f_z[idx] / r_sampling[i];
        v_r[i] = sqrt(v_squared);
    }

    // Return result
    return v_r;
}


std::vector<float> vec_from_array(PyArrayObject *array) {
    // Check that input is a 1-dimensional array of floats
    if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_FLOAT) {
        throw std::invalid_argument("Input must be a 1D NumPy array of floats");
    }

    // Get the size of the array and a pointer to its data
    int size = PyArray_SIZE(array);
    float *data_ptr = static_cast<float *>(PyArray_DATA(array));

    // Create a vector from the array data
    std::vector<float> vec(data_ptr, data_ptr + size);

    return vec;
}


PyArrayObject *array_from_vec(std::vector<float> vec) {
    // Create a 1D NumPy array of the same size as the input vector
    npy_intp size = vec.size();
    PyObject *array = PyArray_SimpleNew(1, &size, NPY_FLOAT);

    // Copy the input vector data to the array data
    float *data_ptr = static_cast<float *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(array)));
    std::memcpy(data_ptr, vec.data(), size * sizeof(float));

    return reinterpret_cast<PyArrayObject *>(array);
}

std::vector<float> costhetaFunc(const std::vector<float> &theta)
{
    unsigned int points = theta.size();
    std::vector<float> res(points);
    for (unsigned int i=0; i<points; i++)
    {
        res[i] = std::cos(theta[i]);
    }
    return res;
}

std::vector<float> sinthetaFunc(const std::vector<float> &theta)
{
    unsigned int points = theta.size();
    std::vector<float> res(points);
    for (unsigned int i = 0; i < points; i++) {
        res[i] = sin(theta[i]);
    }
    return res;
}




std::vector<float> linspace(float start, float end, size_t points)
{
    std::vector<float> res(points);
    float step = (end - start) / (points - 1);
    size_t i = 0;
    for (auto& e : res)
    {
        e = start + step * i++;
    }
    return res;
}

std::vector<float> density(float rho_0, float alpha_0, float rho_1, float alpha_1,std::vector<float> r )
{
    unsigned int vecsize = r.size();
    std::vector<float> density_(vecsize);
    // to kg/lyr^3
    rho_0 *= 1.4171253E27;
    rho_1 *= 1.4171253E27;
    for(unsigned int i = 0; i < vecsize; i++)
    {
        density_[i] = rho_0*exp(-alpha_0*r[i]) + rho_1*exp(-alpha_1*r[i]);
    }
    return density_;
}


class Galaxy {       // The class
public:             // Access specifier
    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    float G;
    float R_max;
    const float pi = 3.141592653589793;
    float dr;
    float alpha_0;
    float rho_0;
    float alpha_1;
    float rho_1;
    float h0;
    float dz;
    float dv0;
    float redshift;
    float GalaxyMass;
    float  sun_mass = 1.9885e30; // mass of the Sun in kilograms
    std::vector<float> r;
    std::vector<float> z;
    std::vector<float> r_sampling;
    std::vector<float> z_sampling;
    std::vector<float> rho;
    std::vector<float> theta;
    std::vector<float> costheta;
    std::vector<float> sintheta;
    std::vector<float> f_z;
    std::vector<float> rotational_velocity_;
    // ######################################
    std::vector<float> x_rotation_points;
    int n_rotation_points = 0;
    std::vector<float> v_rotation_points;
    // ######################################
    bool radial = true;

    Galaxy(float G, float dv0, float rho_0, float alpha_0, float rho_1, float alpha_1, float h0,
           float R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta,
           float redshift = 0.0) {
        // ######################################
        redshift = redshift;
        G = G*(1+redshift);
        r = linspace(1, R_max, nr);
        z = linspace(-h0 / 2.0, h0 / 2.0, nz);
        rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        theta = linspace(0, 2 * pi, ntheta);
        costheta = costhetaFunc(theta);
        sintheta = sinthetaFunc(theta);
        r = linspace(1, R_max, nr);
        z = linspace(-h0 / 2.0, h0 / 2.0, nz);
        rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        theta = linspace(0, 2 * pi, ntheta);
        sintheta = sinthetaFunc(theta);
        costheta = costhetaFunc(theta);
        dv0 = dv0;
        nr_sampling = nr_sampling;
        nz_sampling = nz_sampling;
        // Allocate result vector
        z_sampling = linspace(-h0 / 2.0, h0 / 2.0, nz);
//        z_sampling.push_back(0.0);
        r_sampling = linspace(1,R_max, nr_sampling);
        R_max = R_max;
    }

    std::vector<float> _rotational_velocity() {
        // Allocate result vector
        std::vector<float> z_sampling;
        z_sampling.push_back(0.0);
        std::vector<float> res = calculate_rotational_velocity(G, dv0,
                                                             r_sampling,
                                                             r,
                                                             z,
                                                             costheta,
                                                             sintheta,
                                                             rho,
                                                             redshift);
        return res;
    }


    void find_galaxy_initial_density(const std::vector<float>  vin, std::vector<float> x0){
        rotational_velocity_ = vin;
        float alpha_0 =x0[0];
        float rho_0 = x0[1];
        float alpha_1 = x0[2];
        float rho_1 = x0[3];


    }

    // Define the function to be minimized
    float error_function(const std::vector<float>& x) {
        // Calculate the rotation velocity using the current values of x
        float rho_0 = x[0];
        float alpha_0 = x[1];
        float rho_1 = x[2];
        float alpha_1 = x[3];
        float h0 = x[4];
        // Allocate result vector
        std::vector<float> z_sampling;
        z_sampling.push_back(0.0);
        std::vector<float> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        std::vector<float> vsim = calculate_rotational_velocity(G, dv0,
                                                               x_rotation_points,
                                                               r,
                                                               z,
                                                               costheta,
                                                               sintheta,
                                                               rho,
                                                               redshift); ;
        float error=0.0;
        for (int i=0; i< n_rotation_points; i++){ error += pow( (v_rotation_points[i]-vsim[i]), 2); }
        return error;
    }


// Define the gradient function of the error function
    std::vector<float> gradient(const std::vector<float>& x) {
        const float h = 1e-5;
        const int n = x.size();
        std::vector<float> grad(n);
        float error_0 = error_function(x);
        std::vector<float> xh = x;
        for (int i = 0; i < n; i++) {
            xh[i] += h;
            float error_h = error_function(xh);
            grad[i] = (error_h - error_0) / h;
        }
        return grad;
    }

    void read_galaxy_rotation_curve(std::vector<std::array<float, 2>> vin){
        n_rotation_points = vin.size();
        for (const auto &row : vin) {
            x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
            v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
        }
    }

// Define the gradient descent algorithm
    std::vector<float> gradient_descent(const std::vector<float>& x0, float lr = 0.01, int max_iter = 1000, float eps = 1e-6) {
        std::vector<float> x = x0;
        for (int iter = 0; iter < max_iter; iter++) {
            // Compute the gradient of the error function
            std::vector<float> grad = gradient(x0);
            // Update the parameter values in the direction of steepest descent
            for (int i = 0; i < x.size(); i++) {
                x[i] -= lr * grad[i];
            }
            // Check for convergence
            if (std::abs(grad[0]) < eps && std::abs(grad[1]) < eps && std::abs(grad[2]) < eps && std::abs(grad[3]) < eps) {
                std::cout << "Converged after " << iter << " iterations." << std::endl;
                break;
            }
        }
        return x;
    }


    void set_radial(bool radial){
        radial = radial;
    }

    std::vector<float> get_g(bool radial){
        f_z = get_all_g(G, dv0, r_sampling, z_sampling, r, z,
                             costheta, sintheta, rho, redshift, radial);
        return f_z;
    }

};



int main(){
    std::vector<std::array<float, 2>> m33_rotational_curve = {
            {0.0f, 0.0f},
            {1508.7187f, 38.674137f},
            {2873.3889f, 55.65067f},
            {4116.755f, 67.91063f},
            {5451.099f, 79.22689f},
            {6846.0957f, 85.01734f},
            {8089.462f, 88.38242f},
            {9393.48f, 92.42116f},
            {10727.824f, 95.11208f},
            {11880.212f, 98.342697f},
            {13275.208f, 99.82048f},
            {14609.553f, 102.10709f},
            {18521.607f, 104.25024f},
            {22403.336f, 107.60643f},
            {26406.369f, 115.40966f},
            {30379.076f, 116.87875f},
            {34382.107f, 116.05664f},
            {38354.813f, 117.93005f},
            {42266.87f, 121.42091f},
            {46300.227f, 128.55017f},
            {50212.285f, 132.84966f}
    };

    const float M33_Distance = 3.2E6;
    const float Radius_Universe_4D = 14.03E9;
    float redshift = M33_Distance/(Radius_Universe_4D - M33_Distance);
    const int nr = 100;
    const int nz = 101;
    const int ntheta = 102;
    const int nr_sampling=103;
    const int nz_sampling=104;
    const float G = 7.456866768350099e-46;
    const float R_max = 50000.0;
    const float dr = R_max/nr;
    const float pi = 3.141592653589793;
    const float alpha_0 =0.00042423668409927005;
    const float rho_0 =  12.868348904393013;
    const float alpha_1 = 2.0523892233327836e-05;
    const float rho_1 = 0.13249804158174094;
    const float h0 = 156161.88949004377;
    const float dz = h0/nz;
    const float dtheta = 2*pi/ntheta;
    const float dv0 = dr * dtheta * dz;
    const std::vector<float> r = linspace(1,R_max, nr);
    const std::vector<float> z = linspace(-h0/2.0 , h0/2.0, nz);
    const std::vector<float> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
    const std::vector<float> theta = linspace(0,2*pi, ntheta);
    std::vector<float> f_z = zeros(nr*nz);
    Galaxy M33 = Galaxy(G, dv0, rho_0, alpha_0, rho_1, alpha_1,h0,
                        R_max, nr, nz, nr_sampling, nz_sampling, ntheta,
                        redshift);
    M33.read_galaxy_rotation_curve(m33_rotational_curve);
    std::vector<float> x0 = {alpha_0, rho_0, alpha_1, rho_1, h0};
    M33.gradient_descent(x0);
}
