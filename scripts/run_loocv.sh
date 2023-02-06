#!/bin/bash
#SBATCH --job-name=forestLOOCV
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfeigeli@ucsd.edu
##tempSBATCH --cpus-per-task=8
##removeSBATCH --mem-per-cpu=16G


estimators=$1
python3 loocv.py "$estimators"
