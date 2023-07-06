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
    double redshift = M33_Distance / (Radius_4D - M33_Distance);
    int redshift_birth = 0.0;
    m33_rotational_curve = move_rotation_curve(m33_rotational_curve, redshift, redshift_birth);
    redshift = redshift_birth;
    const int nr = 300;
    const int nz = 100;
    const int ntheta = 180;
    const int nr_sampling = 103;
    const int nz_sampling = 104;
    const double R_max = 50000.0;
    const double GalaxyMass = 5E10;


    const double rho_0 =2.207088e+01; //z=0
    const double alpha_0 =4.832374e-04;
    const double rho_1 =1.212938e-01;
    const double alpha_1 =1.784434e-05;
    const double h0 =1.181379e+05;



//    const double rho_0 =5.973762e+02; // z=2
//    const double alpha_0 =1.474506e-03;
//    const double rho_1 =9.138743e+00;
//    const double alpha_1 =6.146586e-05;
//    const double h0 =4.063806e+04;

//    const double rho_0 =2.576561e+03; //z=4
//    const double alpha_0 =2.357784e-03;
//    const double rho_1 =1.780185e+01;
//    const double alpha_1 =1.283712e-04;
//    const double h0 =2.409097e+04;

//    const double rho_0 =1.251379e+04; //z=8
//    const double alpha_0 =4.270712e-03;
//    const double rho_1 =1.167098e+02;
//    const double alpha_1 =2.674557e-04;
//    const double h0 =1.627423e+04;

//    const double rho_0 =3.185028e+04; //z=12
//    const double alpha_0 =5.735620e-03;
//    const double rho_1 =6.692583e+02;
//    const double alpha_1 =2.510458e-04;
//    const double h0 =1.153284e+04;

//    const double rho_0 =1.412147e+05; //z=18
//    const double alpha_0 =8.936424e-03;
//    const double rho_1 =9.419129e+02;
//    const double alpha_1 =4.800023e-04;
//    const double h0 =6.314548e+03;

//    const double rho_0 =1.473879e+05; //z=20
//    const double alpha_0 =9.645877e-03;
//    const double rho_1 =1.393385e+03;
//    const double alpha_1 =4.403639e-04;
//    const double h0 =7.048836e+03;

//    const double rho_0 =2.092614e+05; // z=22
//    const double alpha_0 =1.080832e-02;
//    const double rho_1 =1.841476e+03;
//    const double alpha_1 =5.600362e-04;
//    const double h0 =6.233305e+03;

//    const double rho_0 =2.620119e+05; //z=24
//    const double alpha_0 =1.176829e-02;
//    const double rho_1 =2.428134e+03;
//    const double alpha_1 =6.140309e-04;
//    const double h0 =5.901798e+03;


//    const double rho_0 =3.666072e+05; //z=27
//    const double alpha_0 =1.312948e-02;
//    const double rho_1 =3.365227e+03;
//    const double alpha_1 =7.009722e-04;
//    const double h0 =5.250272e+03;





    std::vector<double> r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    const std::vector<double> z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    const std::vector<std::vector<double>> rho = density(rho_0, alpha_0, rho_1, alpha_1, r, z);
    const std::vector<double> theta = linspace(0, 2 * M_PI, ntheta);
    bool debug = false;
    const bool cuda= false;
    int GPU_ID = 0;
    double xtol_rel = 1E-6;
    int max_iter = 5000;
    galaxy M33(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
               R_max, nr, nz, nr_sampling, nz_sampling, ntheta, redshift, GPU_ID, cuda, debug, xtol_rel, max_iter );
    M33.read_galaxy_rotation_curve(m33_rotational_curve);



    std::vector<double> x0 = {rho_0, alpha_0, rho_1, alpha_1, h0};
    x0 = {1.844837e+01, 4.740177e-04, 1.457439e-01, 2.269586e-05, 1.360023e+05};
    M33.cuda = true;
    auto velo_1 = M33.simulate_rotation_curve();
    std::vector<double> xout = M33.print_density_parameters();
    print_1D(xout);
    std::cout <<std::endl <<std::endl;
    std::cout << "Redshift is "  << redshift << std::endl << std::endl;
    std::cout << "Total Luminous Mass is "  << calculate_mass(xout[0], xout[1],xout[4]) << std::endl << std::endl;
    std::cout << "Total Gas Mass is "  << calculate_mass(xout[2], xout[3],xout[4]) << std::endl << std::endl;
    std::cout << "const double rho_0 =" << xout[0] <<  ";" << std::endl;
    std::cout << "const double alpha_0 =" << xout[1] <<  ";" <<std::endl;
    std::cout << "const double rho_1 =" << xout[2] <<  ";" <<std::endl;
    std::cout << "const double alpha_1 =" << xout[3] <<  ";" << std::endl;
    std::cout << "const double h0 =" << xout[4] <<  ";" << std::endl;
//    print_1D(velo_1);
//    double new_redshift = 2.0;
//    auto velo_2 = M33.move_galaxy( new_redshift );
//    print_1D(velo_2);
//    int a =1;




//    // Drude Propagation
//    double epoch = 100E6;
//    redshift = Radius_4D/epoch -1;
//    double time_step = 10E6;
//    double eta =100.0;
//    double temperature =7.0;
//    double radius_of_epoch = Radius_4D/(1+redshift);
//    double rho_at_epoch = density_at_cmb*pow(radius_of_cmb/radius_of_epoch,3);
//    std::vector<std::vector<double>> current_masses = M33.DrudePropagator(redshift, time_step, eta, temperature);

}
