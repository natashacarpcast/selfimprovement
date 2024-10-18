#!/bin/bash

#SBATCH --job-name=word2vec
#SBATCH --output=word2vec.out
#SBATCH --error=word2vec.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=160G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 19 --executor-memory 16G --driver-memory 16G lab1.py

# to filter out logging messages from output, run the following:
# cat ssd-spark.out | grep -vE "INFO|WARN"
