#!/bin/bash

#SBATCH --job-name=find-lda
#SBATCH --output=find-lda.out
#SBATCH --error=find-lda.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem=32G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 39 --executor-memory 16G --driver-memory 16G --jars /project/macs40123/spark-jars/spark-nlp_2.12-3.3.2.jar find-lda.py

# to filter out logging messages from output, run the following:
cat find-lda.out | grep -vE "INFO|WARN" > find-lda.txt 