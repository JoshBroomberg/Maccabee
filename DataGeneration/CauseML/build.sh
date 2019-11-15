#! /bin/bash

rm -rf dist build cause_ml.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*
