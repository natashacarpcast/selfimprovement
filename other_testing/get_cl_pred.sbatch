#!/bin/bash

#SBATCH --job-name=get_cl_pred
#SBATCH --output=get_cl_pred.out
#SBATCH --error=get_cl_pred.err
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=16G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 4 --executor-memory 16G --driver-memory 16G get_cl_pred.py

# to filter out logging messages from output, run the following:
cat get_cl_pred.out | grep -vE "INFO|WARN" > get_cl_pred.txt 