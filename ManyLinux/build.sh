cd ~/CLionProjects/HU_GalaxyPackage/
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/home/mp74207/anaconda3/envs/pytorch_env/lib/python3.11/site-packages/torch/share/cmake"  ..
make