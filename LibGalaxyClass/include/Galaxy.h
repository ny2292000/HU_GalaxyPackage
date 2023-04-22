//
// Created by mp74207 on 4/17/23.
//

#ifndef MAINPRONONCUDA_GALAXY_H
#define MAINPRONONCUDA_GALAXY_H

#include <vector>
#include <array>
#include <nlopt.hpp>
#include "../../LibGalaxy/include/lib0.h"

class Galaxy {       // The class
public:             // Access specifier
    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    double R_max;
    double Mtotal_si;
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
    std::vector<std::vector<double>> f_z;
    std::vector<double> rotational_velocity_;
    // ######################################
    std::vector<double> x_rotation_points;
    int n_rotation_points = 0;
    std::vector<double> v_rotation_points;
    // ######################################

    Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0);

    ~Galaxy();

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    get_f_z(const std::vector<double> &x, bool debug=false);


    void read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin);

};



// Define the function to be minimized
static double error_function(const std::vector<double> &x, Galaxy &myGalaxy) {
    // Calculate the rotation velocity using the current values of x
    double rho_0 = x[0];
    double alpha_0 = x[1];
    double rho_1 = x[2];
    double alpha_1 = x[3];
    double h0 = x[4];
    // Calculate the total mass of the galaxy
    double Mtotal_si = massCalc(alpha_0, rho_0, h0);  // Mtotal in Solar Masses
    double error_mass = pow((myGalaxy.GalaxyMass - Mtotal_si) / myGalaxy.GalaxyMass, 2);
    bool debug = false;
    std::vector<double> rho = density(rho_0, alpha_0, rho_1, alpha_1, myGalaxy.r);
    std::vector<double> vsim = calculate_rotational_velocity(myGalaxy.redshift, myGalaxy.dv0,
                                                             myGalaxy.x_rotation_points,
                                                             myGalaxy.r,
                                                             myGalaxy.z,
                                                             myGalaxy.costheta,
                                                             myGalaxy.sintheta,
                                                             rho,
                                                             debug);
    double error = 0.0;
    for (int i = 0; i < myGalaxy.n_rotation_points; i++) { error += pow((myGalaxy.v_rotation_points[i] - vsim[i]), 2); }
    std::cout << "Total Error = " << (error + error_mass) << "\n";
    return error + error_mass;
}

// Define the objective function wrapper
//auto objective_wrapper = [](const std::vector<double>& x, std::vector<double>& grad, void* data);

std::vector<double> nelder_mead(const std::vector<double> &x0, Galaxy &myGalaxy, int max_iter = 1000, double xtol_rel = 1e-6);



#endif //MAINPRONONCUDA_GALAXY_H
