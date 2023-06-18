//============================================================================
// Name        : GalaxyFormation.cpp
// Author      : Marco Pereira
// Version     : 1.0.0
// Copyright   : Your copyright notice
// Description : Hypergeometrical Universe Galaxy Formation in C++, Ansi-style
//============================================================================
#include <vector>
#include <array>
#include "tensor_utils.h"
#include "Galaxy.h"






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


    const double M33_Distance = 3.2E6;
    const double Radius_Universe_4D = 14.03E9;
    double redshift = M33_Distance / (Radius_Universe_4D - M33_Distance);
    const int nr = 300;
    const int nz = 100;
    const int ntheta = 180;
    const int nr_sampling = 103;
    const int nz_sampling = 104;
    const double R_max = 50000.0;
    const double alpha_0 = 0.00042423668409927005;
    const double rho_0 = 12.868348904393013;
    const double alpha_1 = 2.0523892233327836e-05;
    const double rho_1 = 0.13249804158174094;
    const double h0 = 156161.88949004377;
    const double GalaxyMass = 5E10;
    const double pi= 3.141592653589793238;
    //    const std::vector<double> r = linspace(1,R_max, nr);
    std::vector<double> r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    torch::Tensor r_cpu = torch::tensor(r, torch::dtype(torch::kDouble));  // Convert to tensor
    auto mask = (r_cpu < 8089.462);
    int count = mask.sum().item<int>();  // Extract the value as an integer

    const std::vector<double> z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    const std::vector<std::vector<double>> rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    const std::vector<double> theta = linspace(0, 2 * pi, ntheta);
    bool debug = false;
    const bool cuda= false;
    int GPU_ID = 0;
    Galaxy M33(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
               R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift, GPU_ID, cuda, debug);
    M33.read_galaxy_rotation_curve(m33_rotational_curve);
    std::vector<std::vector<double>> f_z = zeros_2(M33.n_rotation_points, 2);
    std::vector<double> x0 = {rho_0, alpha_0, rho_1, alpha_1, h0};
//    auto F_pair = M33.get_f_z(x0, debug);
//    auto f_z_radial = F_pair.first;
//    auto f_z_vertical = F_pair.second;
//    print_2D(f_z_radial);
//    print_2D(f_z_vertical);
//    M33.cuda = true;
//    F_pair = M33.get_f_z(x0, debug);
//    f_z_radial = F_pair.first;
//    f_z_vertical = F_pair.second;
//    print_2D(f_z_radial);
//    print_2D(f_z_vertical);


    x0 = {1.844838e+01, 4.740178e-04, 1.457440e-01, 2.269589e-05, 1.360035e+05};
    M33.cuda = true;
    auto xout = M33.nelder_mead(x0, M33,1000, 1E-3);
    print_1D(xout);

//    bool debug = false;
//    //    std::vector<double> rotational_velocity = calculate_rotational_velocity(redshift, M33.dv0, M33.r_sampling, r, z, M33.costheta, M33.sintheta, M33.rho, debug);
//    std::vector<std::vector<double>> f_z_radial = zeros_2(M33.n_rotation_points, M33.nz);
//    std::vector<std::vector<double>> f_z_vertical = zeros_2(M33.n_rotation_points, M33.nz);
//    auto start = std::chrono::high_resolution_clock::now();
//    auto F_pair = M33.get_f_z(x0, debug);
//    f_z_radial = F_pair.first;
//    f_z_vertical = F_pair.second;
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//    std::cout << duration.count() << std::endl;
//    print_2D(f_z_radial);

}
