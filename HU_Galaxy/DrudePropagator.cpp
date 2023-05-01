#include "Galaxy.h"
#include <vector>
#include <utility>
#include <math.h>


std::vector<std::vector<double>> calculate_tau(double effective_cross_section, const std::vector<std::vector<double>>& local_density, double temperature) {
    // Constants
    const double boltzmann_constant = 1.380649e-23; // J/K
    const double hydrogen_mass = 1.6737236e-27;     // kg

    // Calculate the average velocity of gas particles based on temperature
    double average_velocity = std::sqrt((3 * boltzmann_constant * temperature) / hydrogen_mass);

    // Calculate the number density of gas particles
    std::vector<std::vector<double>> number_density(local_density.size(), std::vector<double>(local_density[0].size(), 0.0));
    for (size_t i = 0; i < local_density.size(); i++) {
        for (size_t j = 0; j < local_density[0].size(); j++) {
            number_density[i][j] = local_density[i][j] / hydrogen_mass;
        }
    }

    // Calculate the time between collisions
    std::vector<std::vector<double>> tau(local_density.size(), std::vector<double>(local_density[0].size(), 0.0));
    for (size_t i = 0; i < local_density.size(); i++) {
        for (size_t j = 0; j < local_density[0].size(); j++) {
            tau[i][j] = 1.0 / (number_density[i][j] * effective_cross_section * average_velocity);
        }
    }

    return tau;
}

//std::pair<int, int> Galaxy::get_center_layers() const {
//    int center_layer1 = nz / 2 - 1;
//    int center_layer2 = nz / 2;
//    return std::make_pair(center_layer1, center_layer2);
//}
//
//// Run the DrudePropagator for some time
//galaxy.DrudePropagator(time_step, eta, temperature);
//
//// Get the indices of the two center layers
//auto center_layers = galaxy.get_center_layers();
//int center_layer1 = center_layers.first;
//int center_layer2 = center_layers.second;
//
//// Calculate the mass inside the two center layers
//double center_mass = 0.0;
//for (int i = 0; i < nr; i++) {
//center_mass += rho[i][center_layer1] * dv0[i];
//center_mass += rho[i][center_layer2] * dv0[i];
//}
//
//// Check if the desired mass has been reached
//if (center_mass >= 0.9 * GalaxyMass) {
//// Galaxy formation complete
//}



void Galaxy::DrudePropagator(double time_step, double eta, double temperature) {
    // Calculate the effective cross-section
    double H_cross_section = 3.53E-20;  //m^2
    double effective_cross_section = eta * H_cross_section;

    // Half of the vertical points
    int half_nz = nz / 2;

    // Calculate the mass in each volume element
    std::vector<std::vector<double>> current_masses(nr, std::vector<double>(nz, 0.0));
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = 0; j < nz; j++) {
            current_masses[i][j] = rho[i][j] * dv0[i];
        }
    }

    // Get the vertical acceleration array
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z =
            get_all_g(redshift, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, recalculate_density(current_masses), false);

    auto z_acceleration = f_z.second;
    std::vector<std::vector<double>> tau = calculate_tau(effective_cross_section, rho, temperature);

    // Loop through radial points
    for (size_t i = 0; i < nr; i++) {
        double dr = (i == 0) ? r[i+1] - r[i] : (r[i] - r[i-1]) / 2.0;
        double dtheta = theta[1] - theta[0];
        double dz = z[1] - z[0];
        double volume_cross_section = r[i] * dtheta * dr * dz;
        // Loop through positive vertical points and zero
        for (size_t j = 1; j <= half_nz; j++) {
            double local_acceleration = z_acceleration[i][j];


            // Calculate drift velocity and mass drift
            double drift_velocity = tau[i][j] * local_acceleration;
            double mass_drift = drift_velocity * time_step * volume_cross_section;

            // Update mass in the current volume
            current_masses[i][j] -= mass_drift;
            current_masses[i][j-1] += mass_drift;
        }

        // Handle mass gain at z=0 (from both sides of the disk)
        double local_acceleration_z0 = z_acceleration[i][0];
        double tau_z0 = tau[i][0];
        double drift_velocity_z0 = tau_z0 * local_acceleration_z0;
        double mass_drift_z0 =  drift_velocity_z0 * time_step * volume_cross_section;
        current_masses[i][0] += mass_drift_z0;
    }

    // Update mass for negative z values
    for (size_t i = 0; i < nr; i++) {
        for (size_t j = half_nz + 1; j < nz; j++) {
            current_masses[i][j] = current_masses[i][nz - j];
        }
    }

    // Recalculate density from updated mass
    rho = recalculate_density(current_masses);
}
