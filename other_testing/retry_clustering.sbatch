#!/bin/bash

#SBATCH --job-name=retry_findk
#SBATCH --output=retry_findk.out
#SBATCH --error=retry_findk.err
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mem=160G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 29 --executor-memory 16G --driver-memory 16G retry_clustering.py

# to filter out logging messages from output, run the following:
cat retry_findk.out | grep -vE "INFO|WARN" > retry_clustering.txt