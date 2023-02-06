import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sys, argparse, random
from fc_helpers import  linearize, get_flat_inds_for_net
from data_loader import data_loader


def prep_data():
    # Load Demographic data and subject lists
    demo = pd.read_csv('../data/demo.csv', names=['VC', 'Age', 'Group'])

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

def run_randomforest_permutation_loocv(X_mat, y, estimators = 500, permutations = 50):

    cv = LeaveOneOut()
    scores = []
    netlist = ['Auditory','CingOperc','CingPar','Default','DorsalAtt','FrontoPar','None', 'RetroTemp','Salience','SMhand','SMmouth','VentralAtt','Visual','Subcort']   
    fold = 0

    # Loop over the LOOCV splits indicies
    for train_ix, test_ix in cv.split(X_mat):
            
        # For each split, create the respective training and test set
        X_train, X_test = X_mat[train_ix, :], X_mat[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # Train the model
        clf = RandomForestClassifier(n_estimators=estimators)

        clf.fit(X_train, y_train)
            
        for net in netlist:
            
            network_inds = get_flat_inds_for_net(net)
    
            temp_test_sub = np.copy(X_test)

            for j in range(permutations):

                # Permute the test subjects network connections individually 
                for i in range(len(network_inds)):
                    randsamp = random.randint(0,X_mat.shape[0]-2)
                    temp_test_sub[0, network_inds[i]] = X_train[randsamp,network_inds[i]]

                # Test the model using the permuted feature set
                loo_score = clf.score(temp_test_sub, y_test)

                # Keep track of the accuracy of the LOOCV with DMN permuted
                scores.append(loo_score)     

                if fold%50 == 0:
                    print(fold)
                fold = fold +1

    return scores        

def main():
    pass

if __name__ == '__main__':
    sys.exit(main())
