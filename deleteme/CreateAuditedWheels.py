import os
import subprocess

# List of Python versions
python_versions = ["cp36-cp36m"]  #, "cp37-cp37m", "cp38-cp38", "cp39-cp39", "cp310-cp310",  "cp311-cp311"  ]

# The path to the Python installations
python_path = "/opt/python"

# The directory where the wheels will be saved
output_dir = "/project/auditedwheels"

# Iterate over each Python version
for version in python_versions:
    # Build the path to the Python binary and pip
    python_bin = os.path.join(python_path, version, "bin", "python")
    pip_bin = os.path.join(python_path, version, "bin", "pip")

    # Create a new virtual environment for this iteration
    venv_path = f"./venv_{version}"
    subprocess.run([python_bin, "-m", "venv", venv_path])
    python_bin_venv = os.path.join(venv_path, "bin", "python")
    pip_bin_venv = os.path.join(venv_path, "bin", "pip")
    subprocess.run([pip_bin_venv, "install", "numpy", "pybind11", "nlopt"])

    # Activate the virtual environment
    venv_bin = os.path.join(venv_path, "bin")
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{env['PATH']}"

    # Build the extension modules
    subprocess.run([python_bin_venv, "setup.py", "build_ext", "--inplace"], env=env)

    # Build the wheel
    subprocess.run([pip_bin_venv, "wheel", ".", "-w", "dist"], env=env)

    # Find the wheel file that was just created
    dist_dir = os.path.join(os.getcwd(), "dist")
    wheel_file = next(f for f in os.listdir(dist_dir) if f.endswith(".whl"))

    # Use auditwheel to repair the wheel
    subprocess.run(["auditwheel", "repair", os.path.join(dist_dir, wheel_file), "-w", output_dir], env=env)

    # Cleanup the dist directory for the next iteration
    subprocess.run(["rm", "-rf", "dist"], env=env)
    subprocess.run(["mkdir", "dist"], env=env)

    # Remove the virtual environment
    subprocess.run(["rm", "-rf", venv_path])
