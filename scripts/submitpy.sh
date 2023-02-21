#!/bin/bash
#SBATCH --job-name=PermuteForest
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfeigeli@ucsd.edu
##cSBATCH --cpus-per-task=8
##cSBATCH --mem-per-cpu=16G

source /sphere/greene-lab/miniconda3/bin/activate greenelab

python3 permute_analysis.py