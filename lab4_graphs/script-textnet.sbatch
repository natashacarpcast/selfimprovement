#!/bin/bash

#SBATCH --job-name=create-network
#SBATCH --output=create-network.out
#SBATCH --error=create-network.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=180G
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python/anaconda-2022.05 spark/3.3.2

export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3

spark-submit --total-executor-cores 9 --executor-memory 18G --driver-memory 18G --jars /project/macs40123/spark-jars/spark-nlp_2.12-3.3.2.jar script-textnet.py

# to filter out logging messages from output, run the following:
cat create-network.out | grep -vE "INFO|WARN" > create-network.txt 