import os
import subprocess

# List of Python versions
python_versions = ["cp36-cp36m", "cp37-cp37m", "cp38-cp38", "cp39-cp39", "cp310-cp310", "cp311-cp311"]

# The path to the Python installations
python_path = "/opt/python"

# The path to the project
project_path = "/path/to/your/project"

# Iterate over each Python version
for version in python_versions:
    # Build the path to the Python binary and pip
    python_bin = os.path.join(python_path, version, "bin", "python")
    pip_bin = os.path.join(python_path, version, "bin", "pip")

    # Create a new virtual environment
    venv_path = os.path.join(python_path, version, "venv")
    subprocess.call([python_bin, "-m", "venv", venv_path])

    # Build the path to the virtual environment's Python binary and pip
    venv_python_bin = os.path.join(venv_path, "bin", "python")
    venv_pip_bin = os.path.join(venv_path, "bin", "pip")

    # Install the necessary packages in the virtual environment
    subprocess.call([venv_pip_bin, "install", "pybind11", "numpy", "pytest==3.1", "wheel", "auditwheel", "twine"])

    # Call cmake to build the package
    os.chdir(project_path)
    subprocess.call(["cmake", "-DPYTHON_EXECUTABLE=" + venv_python_bin, "."])
    subprocess.call(["cmake", "--build", "."])

    # Generate wheel using setup.py
    subprocess.call([venv_python_bin, "setup.py", "bdist_wheel"])

    # Repair the wheel files for compatibility with manylinux
    wheel_directory = os.path.join(project_path, "dist")
    subprocess.call([venv_pip_bin, "install", "auditwheel"])
    subprocess.call([venv_python_bin, "-m", "auditwheel", "repair", "-w", wheel_directory, os.path.join(wheel_directory, "*.whl")])
