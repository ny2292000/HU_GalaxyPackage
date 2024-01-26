source /myvenv/bin/activate
pip uninstall -y hugalaxy
source ./cleanup.sh
CREATE_PACKAGE="TRUE"
export CREATE_PACKAGE
python_env=$(dirname $(dirname $(which python)))
export python_env
echo $python_env
python setup.py sdist bdist_wheel
pip install dist/*.whl
python -c "import hugalaxy as hh; print(hh.__dir__())"
source ./cleanup.sh
unset $CREATE_PACKAGE