import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sys, argparse
from lib import  linearize
from data_loader import data_loader


def prep_data():
    # Load Demographic data and subject lists
    demo = pd.read_csv('data/demo.csv', names=['VC', 'Age', 'Group'])

    demo_ts = demo.where(demo.Group=='TS').dropna().reset_index(drop=True)
    demo_hc = demo.where(demo.Group=='TFC').dropna().reset_index(drop=True)

    # custom data loader reads in connectivity data for each group seperately
    ts_con = data_loader(demo_ts)
    hc_con = data_loader(demo_hc)

    # Connectivity matricies are symettric and square. We need just the flattened upper or lower triangle, of the matrix
    # to create a new design matrix

    ts_con_flat = linearize(ts_con)
    hc_con_flat = linearize(hc_con)

    # create feature matrix
    X = np.vstack((hc_con_flat, ts_con_flat))

    # create label vector: 1 for HC, -1 for TS
    y = np.concatenate((np.repeat(1,99), np.repeat(-1,99)))

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)

    return X_scale, y

def run_randomforest_loocv(X_mat, y, estimators):

    cv = LeaveOneOut()
    scores = []
    cv_fold_importance = []
    fold = 0

    for train_ix, test_ix in cv.split(X_mat):
        
        X_train, X_test = X_mat[train_ix, :], X_mat[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        clf = RandomForestClassifier(n_estimators = estimators)
        clf.fit(X_train, y_train)
        
        cv_fold_importance.append(clf.feature_importances_)
        scores.append(clf.score(X_test, y_test))

        if fold%50 == 0:
            print(fold)
        fold = fold +1
    
    return scores, cv_fold_importance


def main():

    arg_parser = argparse.ArgumentParser()
    if len(sys.argv[1:])==0:
        print('\nArguments required. Use -h option to print FULL usage.\n')
    arg_parser.add_argument('estimators', type=int, help = 'number of random forest estimators')
    args = arg_parser.parse_args()

    X_scale, y = prep_data()
    print(f'running random forest with estimators: {args.estimators}')
    scores, cv_fold_importance = run_randomforest_loocv(X_scale,y,args.estimators)
    np.savetxt(f'loocv_results/scores{args.estimators}.csv', scores, fmt='%.4e', delimiter=',')
    np.savetxt(f'loocv_results/cv_fold_importance{args.estimators}.csv', cv_fold_importance, fmt='%.4e', delimiter=',')

if __name__ == '__main__':
    sys.exit(main())
