import os
import subprocess

def compare_outputs(dir1, dir2):
    """
    Compares the nm and ldd outputs for corresponding files in two directories.

    Args:
      dir1: The path to the first directory.
      dir2: The path to the second directory.
    """

    for filename in os.listdir(dir1):
        if os.path.isfile(os.path.join(dir1, filename)):
            filepath1 = os.path.join(dir1, filename)
            filepath2 = os.path.join(dir2, filename)

            # Get nm output
            nm_output1 = subprocess.getoutput(f"nm {filepath1}").splitlines()
            nm_output2 = subprocess.getoutput(f"nm {filepath2}").splitlines()
            nm_diff = set(nm_output1) - set(nm_output2)

            # Get ldd output
            ldd_output1 = subprocess.getoutput(f"ldd {filepath1}").splitlines()
            ldd_output2 = subprocess.getoutput(f"ldd {filepath2}").splitlines()
            ldd_diff = set(ldd_output1) - set(ldd_output2)

            if nm_diff or ldd_diff:
                print(f"Differences found for {filename}:")
                if nm_diff:
                    print("  nm differences:")
                    for line in nm_diff:
                        print(f"    {line}")
                if ldd_diff:
                    print("  ldd differences:")
                    for line in ldd_diff:
                        print(f"    {line}")

if __name__ == "__main__":
    dir1 = "bin2"
    dir2 = "bin3"
    compare_outputs(dir1, dir2)