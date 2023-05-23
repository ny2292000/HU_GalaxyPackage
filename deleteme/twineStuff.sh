cd /project; rm -rf dist
python setup.py sdist bdist_wheel
cd dist
auditwheel repair /project/dist/HU_Galaxy-0.2-cp310-cp310-linux_x86_64.whl
twine check /project/dist/wheelhouse/*
twine upload --repository-url https://test.pypi.org/legacy/ dist/*





