__global__ void get_all_g_kernel(int nr, int nz, int ntheta, int nr_sampling, int nz_sampling, double G,
                                 const double *r_sampling, const double *z_sampling,
                                 const double *grid_data, const double *rho,
                                 bool radial, double *f_z) {

    // Get the indices of the point in r and z for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Use shared memory for caching intermediate results
    extern __shared__ double shared_data[];

    if (i < nr_sampling && j < nz_sampling) {
        int idx = j + i * nz_sampling;
        // Initialize the result variable for this thread
        double res = 0.0;
        double thisres = 0.0;
        // Loop over all r and z points in the grid
        for (int ir = 0; ir < nr; ir++) {
            if (radial & (grid_data[ir] > r_sampling[i])) {
                break;
            }
            for (int iz = 0; iz < nz; iz++) {
                for (int k = 0; k < ntheta; k++) {
                    // Compute the distance between the sampling point and the point in r and z for this thread
                    double d = pow(z_sampling[j] - grid_data[nr + iz], 2.0) +
                               pow(r_sampling[i] - grid_data[ir] * grid_data[nr + nz + ntheta + k], 2.0) +
                               pow(grid_data[ir] * grid_data[nr + nz + k], 2.0);
                    if (radial) {
                        thisres= G * rho[ir] * grid_data[ir] * grid_data[nr + nz + 2*ntheta + ir] *
                               (r_sampling[i] - grid_data[ir] * grid_data[nr + nz + k]) / pow(d, 1.5);
                        res += thisres;
                    } else {
                        res += G * rho[ir] * grid_data[ir] * grid_data[nr + nz + 2 * ntheta + ir] *
                               (z_sampling[j] - grid_data[nr + iz]) *
                               grid_data[nr + nz + k] / pow(d, 1.5);
                    }
                    if (ir==5 && iz==5 && k == 5) {
                        printf("GPU \n");
                        printf("The value of f_z is %e\n", thisres);
                        printf("The value of distance is %fd\n", sqrt(d));
                        printf("The value of dv0 is %fd\n", grid_data[nr + nz + 2*ntheta + ir]);
                        printf("The value of costheta is %fd\n", grid_data[nr + nz + k]);
                        printf("The value of sintheta is %fd\n", grid_data[nr + nz + ntheta + k]);
                        printf("The value of rsampling is %fd\n", r_sampling[i]);
                        printf("The value of zsampling is %fd\n", z_sampling[j]);
                        printf("The value of rho is %e\n", rho[ir]);
                        printf("The value of G is %e\n", G);
                    }
                }
            }
        }

        // Store the result variable for this thread in the output array
        f_z[idx] = res;
    }
}

std::vector<double> get_all_g_impl_cuda(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
                                        const std::vector<double> &z_sampling, const std::vector<double> &r,
                                        const std::vector<double> &z,
                                        const std::vector<double> &costheta, const std::vector<double> &sintheta,
                                        const std::vector<double> &rho,
                                        bool radial = true) {
    int nr_sampling = r_sampling.size();
    int nz_sampling = z_sampling.size();
    int nr = r.size();
    int nz = z.size();

    // Combine r, z, costheta, sintheta, and dv0 into a single vector (grid_data) for cudaMemcpy
    std::vector<double> grid_data(r);
    grid_data.insert(grid_data.end(), z.begin(), z.end());
    grid_data.insert(grid_data.end(), costheta.begin(), costheta.end());
    grid_data.insert(grid_data.end(), sintheta.begin(), sintheta.end());
    grid_data.insert(grid_data.end(), dv0.begin(), dv0.end());

    // Allocate and copy device memory
    double *f_z = new double(nr_sampling * nz_sampling);
    double *dev_r_sampling = createCUDAVector(r_sampling);
    double *dev_z_sampling = createCUDAVector(z_sampling);
    double *dev_grid_data = createCUDAVector(grid_data);
    double *dev_rho = createCUDAVector(rho);
    double *dev_f_z = createCUDAVector(f_z, nr_sampling * nz_sampling);
    double *dev_dv0 = createCUDAVector(dv0);



// Get the block size for the tensor product kernel
    // Get the block size for the tensor product kernel
    int ntheta = costheta.size();
    int threads_per_block = 32;
//    std::pair<dim3, dim3> block_info = get_block_size(nr * nz * ntheta, threads_per_block);
    std::pair<dim3, dim3> block_info = get_block_size(nr_sampling * nz_sampling, threads_per_block);
    dim3 num_blocks = block_info.first;
    dim3 block_size = block_info.second;


    // Launch kernel
    get_all_g_kernel<<<num_blocks, block_size>>>(nr, nz,ntheta, nr_sampling, nz_sampling, G,
                                                 dev_r_sampling, dev_z_sampling, dev_grid_data, dev_rho,
                                                 radial, dev_f_z);
    // Copy results back to host
    std::vector<double> f_z_vec(nr_sampling * nz_sampling);
    cudaMemcpy(f_z_vec.data(), dev_f_z, nr_sampling * nz_sampling * sizeof(double), cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(dev_r_sampling);
    cudaFree(dev_z_sampling);
    cudaFree(dev_grid_data);
    cudaFree(dev_rho);
    cudaFree(dev_f_z);
    cudaFree(dev_dv0);

    // Return result
    return f_z_vec;
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
    double thisres = 0.0;
    for (unsigned int i = 0; i < nr; i++) {
        if (radial && (r[i] > r_sampling_ii)) {
            break;
        }
        for (unsigned int j = 0; j < nz; j++) {
            for (unsigned int k = 0; k < ntheta; k++) {
                double d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
                           r[i] * r[i] * costheta[k] * costheta[k];
                if (radial) {
                    thisres = G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
                    thisres += res;
                } else {
                    res += G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
                }
                if (i==5 && j==5 && k == 5) {
                    printf("CPU \n");
                    printf("The value of f_z is %e\n\n", thisres);
                    printf("The value of distance is %fd\n", sqrt(d));
                    printf("The value of dv0 is %fd\n", dv0[i]);
                    printf("The value of costheta is %fd\n", costheta[k]);
                    printf("The value of sintheta is %fd\n", sintheta[k]);
                    printf("The value of rsampling is %fd\n", r_sampling_ii);
                    printf("The value of zsampling is %fd\n", z_sampling_jj);
                    printf("The value of rho is %e\n", rho[i]);
                    printf("The value of G is %e\n", G);
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
