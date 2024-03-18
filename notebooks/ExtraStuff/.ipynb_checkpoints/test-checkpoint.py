import os
import numpy as np
import matplotlib.pyplot as plt


def get_files_with_pattern(directory, pattern):
    return [f for f in os.listdir(directory) if pattern in f]

def extract_redshift_from_filename(filename):
    return int(filename.split('_')[-1].split('.')[0])

def sum_masses_from_file(filename):
    data = np.load(filename)
    return np.sum(data, axis=(1,2))  # Summing over the 'radius' and 'elevation' axes

def plot_masses(masses_dict):
    plt.figure(figsize=(12,6))
    for redshift, masses in masses_dict.items():
        plt.plot(masses[0][:],masses[1][:], label=f'Redshift: {redshift}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Total Mass')
    plt.title('Total Mass vs Epoch for different Redshifts')
    plt.show()

def process_and_plot(directory):
    filenames = get_files_with_pattern(directory, '_freefall_all_current_masses_')

    masses_dict = {}  # To store summed masses for each redshift
    for filename in filenames:
        redshift = extract_redshift_from_filename(filename)
        epochs = np.load(directory + "/_freefall_epochs_"+ str(redshift) + '.npy')
        masses = sum_masses_from_file(os.path.join(directory, filename))
        masses_dict[redshift] = (epochs,masses)
    plot_masses(masses_dict)

process_and_plot('/home/mp74207/CLionProjects/HU_GalaxyPackage/notebooks/data')
