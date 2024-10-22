#!/bin/bash

#SBATCH --job-name=mfd2_findk
#SBATCH --output=mfd2_findk.out
#SBATCH --error=mfd2_findk.err
#SBATCH --nodes=1
#SBATCH --ntasks=30
#SBATCH --mem=160G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 29 --executor-memory 16G --driver-memory 16G mfd2_findk.py

# to filter out logging messages from output, run the following:
cat mfd2_findk.out | grep -vE "INFO|WARN"