#!/bin/bash
#SBATCH --job-name=forestLOOCV
###tempremoveSBATCH --cpus-per-task=8
##removeSBATCH --mem-per-cpu=10G

estimators=$1
python3 loocv.py "$estimators"
