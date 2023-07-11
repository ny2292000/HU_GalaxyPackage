//============================================================================
// Name        : GalaxyFormation.cpp
// Author      : Marco Pereira
// Version     : 1.0.0
// Copyright   : Your copyright notice
// Description : Hypergeometrical Universe Galaxy Formation in C++, Ansi-style
//============================================================================
#include <vector>
#include <array>
#include "../src/hugalaxy/tensor_utils.h"
#include "../src/hugalaxy/galaxy.h"
#include "../src/hugalaxy/tensor_utils.h"


std::vector<std::array<double, 2>> interpolate(const std::vector<std::array<double, 2>>& input, size_t num_points) {
    std::vector<std::array<double, 2>> output(num_points);
    double x_min = input[0][0];
    double x_max = input.back()[0];
    double dx = (x_max - x_min) / (num_points - 1);

    for(size_t i = 0; i < num_points; ++i) {
        double x = x_min + dx * i;
        auto it = std::lower_bound(input.begin(), input.end(), std::array<double, 2>{x, 0.0}, [](const std::array<double, 2>& a, const std::array<double, 2>& b){ return a[0] < b[0]; });
        if(it == input.end()) {
            output[i] = {x, input.back()[1]};
        } else if(it == input.begin()) {
            output[i] = {x, input[0][1]};
        } else {
            double x1 = (it-1)->at(0);
            double y1 = (it-1)->at(1);
            double x2 = it->at(0);
            double y2 = it->at(1);
            output[i] = {x, y1 + (y2 - y1) * (x - x1) / (x2 - x1)}; // linear interpolation
        }
    }

    return output;
}




int main() {

    std::vector<std::array<double, 2>> m33_rotational_curve = {
            {0.0f,       0.0f},
            {1508.7187f, 38.674137f},
            {2873.3889f, 55.65067f},
            {4116.755f,  67.91063f},
            {5451.099f,  79.22689f},
            {6846.0957f, 85.01734f},
            {8089.462f,  88.38242f},
            {9393.48f,   92.42116f},
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
            {42266.87f,  121.42091f},
            {46300.227f, 128.55017f},
            {50212.285f, 132.84966f}
    };
    auto interpolated_data = interpolate(m33_rotational_curve, 100);
    const int nr = 100;
    const int nz = 100;
    const int ntheta = 180;
    const int nr_sampling = 103;
    const int nz_sampling = 104;
    const double R_max = 50000.0;
    const double GalaxyMass = 5E10;

    const double M33_Distance = 3.2E6;
    double redshift = M33_Distance / (Radius_4D - M33_Distance);
    double redshift_birth = redshift;
    std::vector<std::array<double,2>> new_rotation_curve = move_rotation_curve(interpolated_data, redshift, redshift_birth );
    // redshift_birth = 34
    std::vector<double> x0 = calculate_density_parameters(redshift_birth);
    print_1D(x0);
    double rho_0 = x0[0]; //z=0
    double alpha_0 = x0[1];
    double rho_1 = x0[2];
    double alpha_1 = x0[3];
    double h0 = x0[4];

    bool debug = false;
    const bool cuda= false;
    int GPU_ID = 0;
    double xtol_rel = 1E-6;
    int max_iter = 5000;
    galaxy M33(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
               R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift_birth, GPU_ID, cuda, debug, xtol_rel, max_iter );
    M33.read_galaxy_rotation_curve(new_rotation_curve);
    std::vector<std::vector<double>> rho = density(rho_0, alpha_0, rho_1, alpha_1, M33.r, M33.z);
//    M33.cuda = false;
//    calculate_rotational_velocity(M33, rho,1.0/alpha_0);
    M33.cuda = true;
//    calculate_rotational_velocity(M33, rho, 1.0/alpha_0);

    auto velo_1 = M33.simulate_rotation_curve();
    std::vector<double> xout = M33.print_density_parameters();
    print_1D(xout);
    std::cout <<std::endl <<std::endl;
    std::cout << "Redshift is "  << M33.redshift << std::endl << std::endl;
    std::cout << "Total Luminous Mass is "  << calculate_mass(xout[0], xout[1],xout[4]) << std::endl << std::endl;
    std::cout << "Total Gas Mass is "  << calculate_mass(xout[2], xout[3],xout[4]) << std::endl << std::endl;
    std::cout << "const double rho_0 =" << xout[0] <<  ";" << std::endl;
    std::cout << "const double alpha_0 =" << xout[1] <<  ";" <<std::endl;
    std::cout << "const double rho_1 =" << xout[2] <<  ";" <<std::endl;
    std::cout << "const double alpha_1 =" << xout[3] <<  ";" << std::endl;
    std::cout << "const double h0 =" << xout[4] <<  ";" << std::endl;





    // Drude Propagation
    double radius_of_epoch = 14.01E9/(1+M33.redshift);
    double density_at_cmb= 1000;
    double time_step = 10E6;
    double eta =100.0;
    double temperature =1.0;
    double radius_of_cmb = 11E6;
    double rho_at_epoch = density_at_cmb*pow(radius_of_cmb/radius_of_epoch,3);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> current_masses = M33.DrudePropagator(M33.redshift, time_step, eta, temperature);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds.\n";
}
