
python setup.py clean --all
python setup.py build_ext --inplace
python setup.py sdist bdist_wheel
twine check dist/*
cp cmake-build-debug/HU_Galaxy/*.so HU_Galaxy/.
cd HU_Galaxy
pip install HU_Galaxy

#twine upload --repository testpypi dist/* --verbose