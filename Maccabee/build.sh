#! /bin/bash

rm -rf dist build maccabee.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*
