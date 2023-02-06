#!/bin/bash
#SBATCH --job-name=forestLOOCV
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfeigeli@ucsd.edu
#SBATCH --output=../results/slurmout/
##cSBATCH --cpus-per-task=8
##cSBATCH --mem-per-cpu=16G

# estimators=$1
# python3 loocv.py "$estimators"

python3 permute_analysis.py  