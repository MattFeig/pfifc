import numpy as np
import sys, argparse
from permute_helpers import prep_data, run_randomforest_permutation_loocv, run_randomforest_permutation_nulls_loocv

def main():

    X_scale, y = prep_data()
    
    print(f'running random forest with permutation analysis')
    # permute_network_scores = run_randomforest_permutation_loocv(X_scale, y)
    permute_nulls_scores = run_randomforest_permutation_nulls_loocv(X_scale, y, n_nulls = 50)

    # np.savetxt(f'../results/permutation_results/permute_network_scores.csv', permute_network_scores, fmt='%.4e', delimiter=',')

    np.savetxt(f'../results/permutation_results/permute_nulls_scores.csv', permute_nulls_scores, fmt='%.4e', delimiter=',')

if __name__ == '__main__':
    sys.exit(main())
