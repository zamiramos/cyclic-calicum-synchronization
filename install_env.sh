#!/bin/bash

PYTHON_ENV='cyclic-synchronization-research-env'

ENVS=$(conda env list | awk '{print $1}' )
if [[ ! $ENVS = *$PYTHON_ENV* ]]; then
    echo conda env "$PYTHON_ENV" is not exist
    conda env create -f environment.yml -n $PYTHON_ENV
else
    echo conda env "$PYTHON_ENV" is already exist
    conda env update -f environment.yml -n $PYTHON_ENV
fi;

source ~/anaconda3/etc/profile.d/conda.sh
conda activate $PYTHON_ENV