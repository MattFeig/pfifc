import numpy as np
import sys, argparse
from permute_helpers import prep_data, run_randomforest_permutation_loocv

def main():

    X_scale, y = prep_data()

    print(f'running random forest with permutation analysis')
    permute_network_scores = run_randomforest_permutation_loocv(X_scale,y)
    np.savetxt(f'../results/permutation_results/permute_network_scores.csv', permute_network_scores, fmt='%.4e', delimiter=',')

if __name__ == '__main__':
    sys.exit(main())
