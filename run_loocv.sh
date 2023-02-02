#!/bin/bash
#SBATCH --job-name=random_forests
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G

estimators=$1
python3 loocv.py "$estimators"