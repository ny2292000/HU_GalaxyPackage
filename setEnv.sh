#!/bin/bash
source ~/.bashrc
eval "$(conda shell.bash hook)"
source /etc/profile
export MODULEPATH=$MODULEPATH:/opt/nvidia/hpc_sdk/modulefiles
module avail
module load nvhpc-hpcx/23.3
module list
conda activate Cosmos