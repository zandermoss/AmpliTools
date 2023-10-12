#! /bin/bash
printf "\n==================[Building AmpliTools]=================\n"
python3 setup.py build
printf "\n=================[Installing AmpliTools]================\n"
pip install -r requirements.txt .
printf "\n====================[Building Docs]=====================\n"
cd docs
make html
cd ..
