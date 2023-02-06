import numpy as np
import sys, argparse
from permute_helpers import prep_data, run_randomforest_loocv

def main():

    arg_parser = argparse.ArgumentParser()
    if len(sys.argv[1:])==0:
        print('\nArguments required. Use -h option to print FULL usage.\n')
    arg_parser.add_argument('estimators', type=int, help = 'number of random forest estimators')
    args = arg_parser.parse_args()

    X_scale, y = prep_data()
    print(f'running random forest with estimators: {args.estimators}')
    scores, cv_fold_importance = run_randomforest_loocv(X_scale,y,args.estimators)

    
    np.savetxt(f'../results/loocv_results/scores{args.estimators}.csv', scores, fmt='%.4e', delimiter=',')
    np.savetxt(f'../results/loocv_results/cv_fold_importance{args.estimators}.csv', cv_fold_importance, fmt='%.4e', delimiter=',')

if __name__ == '__main__':
    sys.exit(main())
