echo " this is the env ${python_env}"
#source /myvenv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
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
#source ./cleanup.sh
unset $CREATE_PACKAGE