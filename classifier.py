from sklearn.model_selection import GridSearchCV
from sklearn import svm
from utils import send_email_notification
import numpy as np
import analyzer
import preprocessing as pre
import data_loader as dl
import feature_extraction as fe
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from analyzer import analyze_feature_performance
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron
from feature_selection import select_k_using_stats
from bci_pipeline import grid_search_cv_linsvm

path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A'

def repeated_k_fold_cv_perceptron(ica_train, train_labels, ica_test, test_labels, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'l1', feature_selection_methods = [], features = [],
                           alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100], dur = 2, standardise_features = True):
    if repeats>1:
        kf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    else:
        kf = StratifiedKFold(n_splits = folds, shuffle = True)
        
    if len(feature_selection_methods) == 0:
        features = [ica_train.shape[0]*40]
        feature_selection_methods = ['None']
        do_fs = False
    else:
        if len(features) == 0:
            features = [10, 30, 50, 100, 250, 500, 750]
        do_fs = True
        #features = np.arange(30, 50, 2)

    ar_orders = np.arange(5, 80, 10)
    ar_orders = [10, 25, 50, 75, 100]
    feature_extraction_params = ['boxcar'] if feature_extraction_method == 'Periodogram_PSD' else ar_orders
    fe_param_label = "window" if feature_extraction_method == 'Periodogram_PSD' else 'AR_model_order'

    cv_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(alpha), folds*repeats))
    test_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(alpha), folds*repeats))
    
    df_cv = pd.DataFrame(columns = ['Classifier', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'Alpha', 'Avg CV Acc', 'Var CV Acc'])
    df_test = pd.DataFrame(columns = ['Classifier', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'Alpha', 'Avg Test Acc'])
    split_count = 0
    progress_count = 0
    total = folds*repeats*len(feature_extraction_params)*len(features)*len(feature_selection_methods)*len(alpha)
    for train_idx, cv_idx in kf.split(np.zeros(len(train_labels)), train_labels):
        print("working on split {}/{}".format(split_count+1, folds))
        for fe_param_idx, fe_param in enumerate(feature_extraction_params):
            print("working on fe param {}/{}".format(fe_param_idx+1, len(feature_extraction_params)))
            extra_args = {fe_param_label:fe_param}
            X_train, y_train, freq = fe.extract_psd_features(train_labels, ica_train, feature_extraction_method, extra_args, window_duration = dur)
            X_test, y_test, freq = fe.extract_psd_features(test_labels, ica_test, feature_extraction_method, extra_args, window_duration = dur)
            
            X_train_cv, X_test_cv = X_train[train_idx, :], X_train[cv_idx, :]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[cv_idx] 
            for feature_idx, feature_count in enumerate(features):
                print("working on fea select count {}/{}".format(feature_idx+1, len(features)))
                for feature_select_idx, feature_select_method in enumerate(feature_selection_methods):
                    print("working on fs method count {}/{}".format(feature_select_idx+1, len(feature_selection_methods)))
                    if do_fs: 
                        X_train_cv_red, selector = select_k_using_stats(X_train_cv, y_train_cv, feature_count, metric = feature_select_method)
                        X_test_cv_red = selector.transform(X_test_cv)
                        
                        X_train_red, selector_full = select_k_using_stats(X_train, y_train, feature_count, metric = feature_select_method)
                        X_test_red = selector_full.transform(X_test)
                    else:
                        X_train_cv_red = X_train_cv
                        X_test_cv_red = X_test_cv
                        X_train_red = X_train
                        X_test_red = X_test
                        
                    if standardise_features:
                        X_train_cv_stand, scaler_cv = pre.standardise_data(X_train_cv_red)
                        X_test_cv_stand = scaler_cv.transform(X_test_cv_red)
                        
                        X_train_stand, scaler = pre.standardise_data(X_train_red)
                        X_test_stand = scaler.transform(X_test_red)
                    else: 
                        X_train_cv_stand = X_train_cv_red
                        X_test_cv_stand = X_test_cv_red
                        X_train_stand = X_train_red
                        X_test_stand = X_test_red
                    
                    for c_idx, c in enumerate(alpha):
                        progress_count += 1
                        print("working on alpha: {}/{}".format(c_idx+1, len(alpha)))
                    
                        perceptron = train_perceptron(X_train_cv_stand, y_train_cv, penalty, c)
                        cv_acc = perceptron.score(X_test_cv_stand, y_test_cv)
                        cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = cv_acc
                        perceptron = train_perceptron(X_train_stand, y_train, penalty, c)
                        test_acc = perceptron.score(X_test_stand, y_test)
                        test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = test_acc
                        print("{:.2%} finished".format(progress_count/total))
                        if (split_count+1) == (folds*repeats):
                            cv_acc = cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            test_acc = test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            df_cv = df_cv.append({'Classifier': 'Perceptron', 'Penalty':penalty, 'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'Alpha':c, 'Avg CV Acc':cv_acc.mean(), 
                                                        'Var CV Acc': cv_acc.var()}, ignore_index = True)
    
                            df_test = df_test.append({'Classifier': 'Perceptron', 'Penalty':penalty,'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'Alpha':c, 'Avg Test Acc':test_acc.mean()}, ignore_index = True)
        
   
        split_count +=1                        
    return df_cv, df_test, cv_accs, test_accs

def repeated_k_fold_cv_logreg(ica_train, train_labels, ica_test, test_labels, duals = False, solver = 'liblinear', folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = False, feature_selection_methods = [], features = [],
                           C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100], dur = 2):
    if repeats>1:
        kf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    else:
        kf = StratifiedKFold(n_splits = folds, shuffle = True)
        
    if len(feature_selection_methods) == 0:
        features = [ica_train.shape[0]*40]
        feature_selection_methods = ['None']
        do_fs = False
    else:
        if len(features) == 0:
            features = [10, 30, 50, 100, 250, 500, 750]
        do_fs = True
        #features = np.arange(30, 50, 2)

    ar_orders = np.arange(5, 80, 10)
    ar_orders = [10, 25, 50, 75, 100]
    feature_extraction_params = ['boxcar'] if feature_extraction_method == 'Periodogram_PSD' else ar_orders
    fe_param_label = "window" if feature_extraction_method == 'Periodogram_PSD' else 'AR_model_order'

    cv_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(C), folds*repeats))
    test_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(C), folds*repeats))
    
    df_cv = pd.DataFrame(columns = ['Classifier', 'Solver', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'C', 'Avg CV Acc', 'Var CV Acc'])
    df_test = pd.DataFrame(columns = ['Classifier', 'Solver', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'C', 'Avg Test Acc'])
    split_count = 0
    progress_count = 0
    total = folds*repeats*len(feature_extraction_params)*len(features)*len(feature_selection_methods)*len(C)
    for train_idx, cv_idx in kf.split(np.zeros(len(train_labels)), train_labels):
        print("working on split {}/{}".format(split_count+1, folds))
        for fe_param_idx, fe_param in enumerate(feature_extraction_params):
            print("working on fe param {}/{}".format(fe_param_idx+1, len(feature_extraction_params)))
            extra_args = {fe_param_label:fe_param}
            X_train, y_train, freq = fe.extract_psd_features(train_labels, ica_train, feature_extraction_method, extra_args, window_duration = dur)
            X_test, y_test, freq = fe.extract_psd_features(test_labels, ica_test, feature_extraction_method, extra_args, window_duration = dur)
            
            X_train_cv, X_test_cv = X_train[train_idx, :], X_train[cv_idx, :]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[cv_idx] 
            for feature_idx, feature_count in enumerate(features):
                print("working on fea select count {}/{}".format(feature_idx+1, len(features)))
                for feature_select_idx, feature_select_method in enumerate(feature_selection_methods):
                    print("working on fs method count {}/{}".format(feature_select_idx+1, len(feature_selection_methods)))
                    if do_fs: 
                        X_train_cv_red, selector = select_k_using_stats(X_train_cv, y_train_cv, feature_count, metric = feature_select_method)
                        X_test_cv_red = selector.transform(X_test_cv)
                        
                        X_train_red, selector_full = select_k_using_stats(X_train, y_train, feature_count, metric = feature_select_method)
                        X_test_red = selector_full.transform(X_test)
                    else:
                        X_train_cv_red = X_train_cv
                        X_test_cv_red = X_test_cv
                        X_train_red = X_train
                        X_test_red = X_test
                        
                    X_train_cv_stand, scaler_cv = pre.standardise_data(X_train_cv_red)
                    X_test_cv_stand = scaler_cv.transform(X_test_cv_red)
                    
                    X_train_stand, scaler = pre.standardise_data(X_train_red)
                    X_test_stand = scaler.transform(X_test_red)
                    
                    for c_idx, c in enumerate(C):
                        progress_count += 1
                        print("working on c: {}/{}".format(c_idx+1, len(C)))
                        penalty = 'l1' if is_l1 else 'l2'

                        lr = train_logreg(X_train_cv_stand, y_train_cv, penalty, c, duals = duals, solver = solver)
                        cv_acc = lr.score(X_test_cv_stand, y_test_cv)
                        cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = cv_acc
                        lr = train_logreg(X_train_stand, y_train, penalty, c, duals = duals, solver = solver)
                        test_acc = lr.score(X_test_stand, y_test)
                        test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = test_acc
                        print("{:.2%} finished".format(progress_count/total))
                        if (split_count+1) == (folds*repeats):
                            cv_acc = cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            test_acc = test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            df_cv = df_cv.append({'Classifier': 'Log Reg', 'Solver': solver, 'Penalty':penalty, 'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'C':c, 'Avg CV Acc':cv_acc.mean(), 
                                                        'Var CV Acc': cv_acc.var()}, ignore_index = True)
    
                            df_test = df_test.append({'Classifier': 'Log Reg', 'Solver': solver, 'Penalty':penalty,'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'C':c, 'Avg Test Acc':test_acc.mean()}, ignore_index = True)
        
   
        split_count +=1                        
    return df_cv, df_test, cv_accs, test_accs

def repeated_k_fold_cv_linsvm(ica_train, train_labels, ica_test, test_labels, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = True, feature_selection_methods = [], features = [],
                           C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100], dur = 2):
    if repeats>1:
        kf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    else:
        kf = StratifiedKFold(n_splits = folds, shuffle = True)
        
    if len(feature_selection_methods) == 0:
        features = [ica_train.shape[0]*40]
        feature_selection_methods = ['None']
        do_fs = False
    else:
        if len(features) == 0:
            features = [10, 30, 50, 100, 250, 500, 750]
        do_fs = True
        #features = np.arange(30, 50, 2)

    ar_orders = np.arange(5, 80, 10)
    ar_orders = [10, 25, 50, 75, 100]
    feature_extraction_params = ['boxcar'] if feature_extraction_method == 'Periodogram_PSD' else ar_orders
    fe_param_label = "window" if feature_extraction_method == 'Periodogram_PSD' else 'AR_model_order'

    cv_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(C), folds*repeats))
    test_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(C), folds*repeats))
    
    df_cv = pd.DataFrame(columns = ['Classifier', 'Loss Fxn', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'C', 'Avg CV Acc', 'Var CV Acc'])
    df_test = pd.DataFrame(columns = ['Classifier', 'Loss Fxn', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'C', 'Avg Test Acc'])
    split_count = 0
    progress_count = 0
    total = folds*repeats*len(feature_extraction_params)*len(features)*len(feature_selection_methods)*len(C)
    for train_idx, cv_idx in kf.split(np.zeros(len(train_labels)), train_labels):
        print("working on split {}/{}".format(split_count+1, folds))
        for fe_param_idx, fe_param in enumerate(feature_extraction_params):
            print("working on fe param {}/{}".format(fe_param_idx+1, len(feature_extraction_params)))
            extra_args = {fe_param_label:fe_param}
            X_test, y_test, freq = fe.extract_psd_features(test_labels, ica_test, feature_extraction_method, extra_args, window_duration = dur)
            
            X_train_cv, X_test_cv = X_train[train_idx, :], X_train[cv_idx, :]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[cv_idx] 
            for feature_idx, feature_count in enumerate(features):
                print("working on fea select count {}/{}".format(feature_idx+1, len(features)))
                for feature_select_idx, feature_select_method in enumerate(feature_selection_methods):
                    print("working on fs method count {}/{}".format(feature_select_idx+1, len(feature_selection_methods)))
                    if do_fs: 
                        X_train_cv_red, selector = select_k_using_stats(X_train_cv, y_train_cv, feature_count, metric = feature_select_method)
                        X_test_cv_red = selector.transform(X_test_cv)
                        
                        X_train_red, selector_full = select_k_using_stats(X_train, y_train, feature_count, metric = feature_select_method)
                        X_test_red = selector_full.transform(X_test)
                    else:
                        X_train_cv_red = X_train_cv
                        X_test_cv_red = X_test_cv
                        X_train_red = X_train
                        X_test_red = X_test
                        
                    X_train_cv_stand, scaler_cv = pre.standardise_data(X_train_cv_red)
                    X_test_cv_stand = scaler_cv.transform(X_test_cv_red)
                    
                    X_train_stand, scaler = pre.standardise_data(X_train_red)
                    X_test_stand = scaler.transform(X_test_red)
                    
                    for c_idx, c in enumerate(C):
                        progress_count += 1
                        print("working on c: {}/{}".format(c_idx+1, len(C)))
                        penalty = 'l1' if is_l1 else 'l2'
                        dual_form = not is_l1
                        lin_svm = train_linear_SVM(X_train_cv_stand, y_train_cv, 'squared_hinge', penalty, c, duals = dual_form)
                        cv_acc = lin_svm.score(X_test_cv_stand, y_test_cv)
                        cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = cv_acc
                        lin_svm = train_linear_SVM(X_train_stand, y_train, 'squared_hinge', penalty, c, duals = dual_form)
                        test_acc = lin_svm.score(X_test_stand, y_test)
                        test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = test_acc
                        print("{:.2%} finished".format(progress_count/total))
                        if (split_count+1) == (folds*repeats):
                            cv_acc = cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            test_acc = test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            df_cv = df_cv.append({'Classifier': 'Linear SVM', 'Loss Fxn': 'squared hinge', 'Penalty':penalty,'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'C':c, 'Avg CV Acc':cv_acc.mean(), 
                                                        'Var CV Acc': cv_acc.var()}, ignore_index = True)
    
                            df_test = df_test.append({'Classifier': 'Linear SVM', 'Loss Fxn': 'squared hinge', 'Penalty':penalty,'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'C':c, 'Avg Test Acc':test_acc.mean()}, ignore_index = True)
        
   
        split_count +=1                        
    return df_cv, df_test, cv_accs, test_accs
'''
def repeated_k_fold_cv_kernelsvm(ica_train, train_labels, ica_test, test_labels, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       ,  feature_selection_methods = [], kernel = 'RBF', gamma = ['scale', 'auto']
                                                      , C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100], dur = 2):
    if repeats>1:
        kf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    else:
        kf = StratifiedKFold(n_splits = folds, shuffle = True)
        
    if len(feature_selection_methods) == 0:
        features = [ica_train.shape[0]*40]
        feature_selection_methods = ['None']
        do_fs = False
    else:
        if len(features) == 0:
            features = [10, 30, 50, 100, 250, 500, 750]
        do_fs = True
        #features = np.arange(30, 50, 2)

    ar_orders = np.arange(5, 80, 10)
    ar_orders = [10, 25, 50, 75, 100]
    feature_extraction_params = ['boxcar'] if feature_extraction_method == 'Periodogram_PSD' else ar_orders
    fe_param_label = "window" if feature_extraction_method == 'Periodogram_PSD' else 'AR_model_order'

    cv_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), len(gamma), len(C), folds*repeats))
    test_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), len(gamma) , len(C), folds*repeats))
    
    df_cv = pd.DataFrame(columns = ['Classifier', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'Gamma', 'C', 'Avg CV Acc', 'Var CV Acc'])
    df_test = pd.DataFrame(columns = ['Classifier', 'Loss Fxn', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'Gamma', 'C', 'Avg Test Acc'])
    split_count = 0
    progress_count = 0
    total = folds*repeats*len(feature_extraction_params)*len(features)*len(feature_selection_methods)*len(C)*len(gamma)
    for train_idx, cv_idx in kf.split(np.zeros(len(train_labels)), train_labels):
        print("working on split {}/{}".format(split_count+1, folds))
        for fe_param_idx, fe_param in enumerate(feature_extraction_params):
            print("working on fe param {}/{}".format(fe_param_idx+1, len(feature_extraction_params)))
            extra_args = {fe_param_label:fe_param}
            X_train, y_train, freq = fe.extract_psd_features(train_labels, ica_train, feature_extraction_method, extra_args, window_duration = dur)
            X_test, y_test, freq = fe.extract_psd_features(test_labels, ica_test, feature_extraction_method, extra_args, window_duration = dur)
            
            X_train_cv, X_test_cv = X_train[train_idx, :], X_train[cv_idx, :]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[cv_idx] 
            for feature_idx, feature_count in enumerate(features):
                print("working on fea select count {}/{}".format(feature_idx+1, len(features)))
                for feature_select_idx, feature_select_method in enumerate(feature_selection_methods):
                    print("working on fs method count {}/{}".format(feature_select_idx+1, len(feature_selection_methods)))
                    if do_fs: 
                        X_train_cv_red, selector = select_k_using_stats(X_train_cv, y_train_cv, feature_count, metric = feature_select_method)
                        X_test_cv_red = selector.transform(X_test_cv)
                        
                        X_train_red, selector_full = select_k_using_stats(X_train, y_train, feature_count, metric = feature_select_method)
                        X_test_red = selector_full.transform(X_test)
                    else:
                        X_train_cv_red = X_train_cv
                        X_test_cv_red = X_test_cv
                        X_train_red = X_train
                        X_test_red = X_test
                        
                    X_train_cv_stand, scaler_cv = pre.standardise_data(X_train_cv_red)
                    X_test_cv_stand = scaler_cv.transform(X_test_cv_red)
                    
                    X_train_stand, scaler = pre.standardise_data(X_train_red)
                    X_test_stand = scaler.transform(X_test_red)
                    
                    for gamma_idx, gamma_value in enumerate(gamma):
                        for c_idx, c in enumerate(C):
                            progress_count += 1
                            print("working on c: {}/{}".format(c_idx+1, len(C)))
                            
                            svm = train_kernel_SVM(X_train_cv_stand, y_train_cv, 'squared_hinge', penalty, c, duals = dual_form)
                            cv_acc = lin_svm.score(X_test_cv_stand, y_test_cv)
                            cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = cv_acc
                            lin_svm = train_linear_SVM(X_train_stand, y_train, 'squared_hinge', penalty, c, duals = dual_form)
                            test_acc = lin_svm.score(X_test_stand, y_test)
                            test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = test_acc
                            print("{:.2%} finished".format(progress_count/total))
                            if (split_count+1) == (folds*repeats):
                                cv_acc = cv_accs[fe_param_idx, feature_idx, feature_select_idx, gamma_idx, c_idx, :]
                                test_acc = test_accs[fe_param_idx, feature_idx, feature_select_idx, gamma_idx, c_idx, :]
                                df_cv = df_cv.append({'Classifier': 'Linear SVM', 'Loss Fxn': 'squared hinge', 'Penalty':penalty,'Feature Type': feature_extraction_method, 
                                                    fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                        'Feature Select Metric':feature_select_method, 'C':c, 'Avg CV Acc':cv_acc.mean(), 
                                                            'Var CV Acc': cv_acc.var()}, ignore_index = True)
        
                                df_test = df_test.append({'Classifier': 'Linear SVM', 'Loss Fxn': 'squared hinge', 'Penalty':penalty,'Feature Type': feature_extraction_method, 
                                                    fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                        'Feature Select Metric':feature_select_method, 'C':c, 'Avg Test Acc':test_acc.mean()}, ignore_index = True)
            
   
        split_count +=1                        
    return df_cv, df_test, cv_accs, test_accs
'''
# QDA and LDA
def repeated_k_fold_cv_DA(ica_train, train_labels, ica_test, test_labels, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_lda = True, solver = 'svd', reg_params = [None], feature_selection_methods = [], features = [], dur = 2):
    if repeats>1:
        kf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    else:
        kf = StratifiedKFold(n_splits = folds, shuffle = True)
        
    if len(feature_selection_methods) == 0:
        features = [ica_train.shape[0]*40]
        feature_selection_methods = ['None']
        do_fs = False
    else:
        if len(features) == 0:
            features = [10, 30, 50, 100, 250, 500, 750]
        do_fs = True
        #features = np.arange(30, 50, 2)        
    ar_orders = np.arange(5, 55, 5)
    
    feature_extraction_params = ['boxcar'] if feature_extraction_method == 'Periodogram_PSD' else ar_orders
    fe_param_label = "window" if feature_extraction_method == 'Periodogram_PSD' else 'ar_model'

    cv_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(reg_params), folds*repeats))
    test_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, len(reg_params), folds*repeats))
    
    df_cv = pd.DataFrame(columns = ['Classifier', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'Shrinkage/Reg param', 'Avg CV Acc', 'Var CV Acc'])
    df_test = pd.DataFrame(columns = ['Classifier', 'Loss Fxn', 'Penalty', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'Shrinkage/Reg param', 'Avg Test Acc'])
    split_count = 0
    progress_count = 0
    total = folds*repeats*len(feature_extraction_params)*len(features)*len(feature_selection_methods)*len(reg_params)
    for train_idx, cv_idx in kf.split(np.zeros(len(train_labels)), train_labels):
        for fe_param_idx, fe_param in enumerate(feature_extraction_params):
            extra_args = {fe_param_label:fe_param}
            X_train, y_train, freq = fe.extract_psd_features(train_labels, ica_train, feature_extraction_method, extra_args, window_duration = dur)
            X_test, y_test, freq = fe.extract_psd_features(test_labels, ica_test, feature_extraction_method, extra_args, window_duration = dur)
            
            X_train_cv, X_test_cv = X_train[train_idx, :], X_train[cv_idx, :]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[cv_idx] 
            for feature_idx, feature_count in enumerate(features):
                for feature_select_idx, feature_select_method in enumerate(feature_selection_methods):
                    if do_fs: 
                        X_train_cv_red, selector = select_k_using_stats(X_train_cv, y_train_cv, feature_count, metric = feature_select_method)
                        X_test_cv_red = selector.transform(X_test_cv)
                        
                        X_train_red, selector_full = select_k_using_stats(X_train, y_train, feature_count, metric = feature_select_method)
                        X_test_red = selector_full.transform(X_test)
                    else:
                        X_train_cv_red = X_train_cv
                        X_test_cv_red = X_test_cv
                        X_train_red = X_train
                        X_test_red = X_test
                    
                    X_train_cv_stand, scaler_cv = pre.standardise_data(X_train_cv_red)
                    X_test_cv_stand = scaler_cv.transform(X_test_cv_red)
                    
                    X_train_stand, scaler = pre.standardise_data(X_train_red)
                    X_test_stand = scaler.transform(X_test_red)
                    
                    for c_idx, c in enumerate(reg_params):
                        progress_count += 1
                        print("{:.2%} finished".format(progress_count/total))
                        
                        if is_lda:
                            classifier_cv = train_LDA(X_train_cv_stand, y_train_cv, solver = solver, shrinkage = c)
                            classifier_full = train_LDA(X_train_stand, y_train, solver = solver, shrinkage = c)
                        else:
                            classifier_cv = train_QDA(X_train_cv_red, y_train_cv, reg_param = c)
                            classifier_full = train_QDA(X_train_stand, y_train, reg_param = c)

                        cv_acc = classifier_cv.score(X_test_cv_stand, y_test_cv)
                        cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = cv_acc

                        test_acc = classifier_full.score(X_test_stand, y_test)
                        test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, split_count] = test_acc
                        
                        if (split_count+1) == (folds*repeats):
                            cv_acc = cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            test_acc = test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, c_idx, :]
                            df_cv = df_cv.append({'Classifier': 'LDA' if is_lda else 'QDA', 'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'Shrinkage/Reg param':c, 'Avg CV Acc':cv_acc.mean(), 
                                                        'Var CV Acc': cv_acc.var()}, ignore_index = True)
    
                            df_test = df_test.append({'Classifier': 'LDA' if is_lda else 'QDA', 'Feature Type': feature_extraction_method, 
                                                fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                    'Feature Select Metric':feature_select_method, 'Shrinkage/Reg param':c, 'Avg Test Acc':test_acc.mean()}, ignore_index = True)
        
   
        split_count +=1                        
    return df_cv, df_test, cv_accs, test_accs

# Naive Bayes
def repeated_k_fold_cv_NB(ica_train, train_labels, test_labels, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , feature_selection_methods = [], features = [], dur = 2):
    if repeats>1:
        kf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    else:
        kf = StratifiedKFold(n_splits = folds, shuffle = True)
        
    if len(feature_selection_methods) == 0:
        features = [ica_train.shape[0]*40]
        feature_selection_methods = ['None']
        do_fs = False
    else:
        if len(features) == 0:
            features = [30, 50, 100, 250, 500, 750]
        do_fs = True
        #features = np.arange(30, 50, 2)        
    ar_orders = np.arange(5, 55, 5)
    
    feature_extraction_params = ['boxcar'] if feature_extraction_method == 'Periodogram_PSD' else ar_orders
    fe_param_label = "window" if feature_extraction_method == 'Periodogram_PSD' else 'ar_model'

    cv_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1, folds*repeats))
    test_accs = np.zeros((len(feature_extraction_params), len(features), len(feature_selection_methods), 1,  folds*repeats))
    
    df_cv = pd.DataFrame(columns = ['Classifier', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric', 'Avg CV Acc', 'Var CV Acc'])
    df_test = pd.DataFrame(columns = ['Classifier', 'Feature Type', fe_param_label, 'Feature Count', 
                                     'Feature Select Metric','Avg Test Acc'])
    split_count = 0
    progress_count = 0
    for train_idx, cv_idx in kf.split(np.zeros(len(train_labels)), train_labels):
        for fe_param_idx, fe_param in enumerate(feature_extraction_params):
            extra_args = {fe_param_label:fe_param}
            X_train, y_train, freq = fe.extract_psd_features(train_labels, ica_train, feature_extraction_method, extra_args, window_duration = dur)
            X_test, y_test, freq = fe.extract_psd_features(test_labels, ica_test, feature_extraction_method, extra_args, window_duration = dur)
            
            X_train_cv, X_test_cv = X_train[train_idx, :], X_train[cv_idx, :]
            y_train_cv, y_test_cv = y_train[train_idx], y_train[cv_idx] 
            for feature_idx, feature_count in enumerate(features):
                for feature_select_idx, feature_select_method in enumerate(feature_selection_methods):
                    if do_fs: 
                        X_train_cv_red, selector = select_k_using_stats(X_train_cv, y_train_cv, feature_count, metric = feature_select_method)
                        X_test_cv_red = selector.transform(X_test_cv)
                        
                        X_train_red, selector_full = select_k_using_stats(X_train, y_train, feature_count, metric = feature_select_method)
                        X_test_red = selector_full.transform(X_test)
                    else:
                        X_train_cv_red = X_train_cv
                        X_test_cv_red = X_test_cv
                        X_train_red = X_train
                        X_test_red = X_test
                    
                    X_train_cv_stand, scaler_cv = pre.standardise_data(X_train_cv_red)
                    X_test_cv_stand = scaler_cv.transform(X_test_cv_red)
                    
                    X_train_stand, scaler = pre.standardise_data(X_train_red)
                    X_test_stand = scaler.transform(X_test_red)
            
                    progress_count += 1
                    total = folds*repeats*len(feature_extraction_params)*len(features)*len(feature_selection_methods)
                    print("{:.2%} finished".format(progress_count/total))
                    
                    classifier_cv = train_NB(X_train_cv_stand, y_train_cv)
                    cv_acc = classifier_cv.score(X_test_cv_stand, y_test_cv)
                    cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, split_count] = cv_acc
                    
                    classifier_full = train_NB(X_train_stand, y_train)
                    test_acc = classifier_full.score(X_test_stand, y_test)
                    test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, split_count] = test_acc
                    
                    if (split_count+1) == (folds*repeats):
                        cv_acc = cv_accs[fe_param_idx, feature_idx, feature_select_idx, 0, :]
                        test_acc = test_accs[fe_param_idx, feature_idx, feature_select_idx, 0, :]
                        df_cv = df_cv.append({'Classifier': 'NB', 'Feature Type': feature_extraction_method, 
                                            fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                'Feature Select Metric':feature_select_method, 'Avg CV Acc':cv_acc.mean(), 
                                                    'Var CV Acc': cv_acc.var()}, ignore_index = True)

                        df_test = df_test.append({'Classifier': 'NB', 'Feature Type': feature_extraction_method, 
                                            fe_param_label:fe_param, 'Feature Count': feature_count, 
                                                'Feature Select Metric':feature_select_method, 'Avg Test Acc':test_acc.mean()}, ignore_index = True)
        
   
        split_count +=1                        
    return df_cv, df_test, cv_accs, test_accs

def kfold_cv(classifier, X_train, y_train, param_grid, send_notif, title, folds=10):
    grid_search = GridSearchCV(classifier, param_grid, cv=folds)
    grid_search.fit(X_train, y_train)
    analyzer.display_score_matrix(grid_search)
    if send_notif: 
        send_email_notification("{}\n\nResults for search: {}".format(title, analyzer.get_string_results(grid_search)))
    return grid_search

def find_linear_SVM(X_train, y_train, param_grid, pen, loss_fxn, send_notif, title, folds=10, duals = True):
    X_train, scaler = pre.standardise_data(X_train)
    lin_svm = svm.LinearSVC(penalty=pen, loss=loss_fxn, dual=duals, max_iter = 100000000)
    print(lin_svm)
    grid_search = kfold_cv(lin_svm, X_train, y_train, param_grid, send_notif, title, folds)
    return grid_search.cv_results_['mean_test_score'], grid_search.best_params_
    
def train_linear_SVM(X_train, y_train, loss_fxn, pen, c, duals = True):
    lin_svm = svm.LinearSVC(penalty = pen, loss = loss_fxn, dual = duals, C = c, max_iter = 100000000)
    lin_svm.fit(X_train, y_train)
    return lin_svm

def train_kernel_SVM(X_train, y_train, kernel, gamma, c):
    svm = svm.SVC(C = c, kernel = kernel, gamma = gamma)
    svm.fit(X_train, y_train)
    return svm

def train_LDA(X_train, y_train, solver = 'svd', shrinkage = None):
    lda = LinearDiscriminantAnalysis(solver = solver, shrinkage = shrinkage)
    lda.fit(X_train, y_train)
    return lda

def train_QDA(X_train, y_train, reg_param = None):
    if reg_param == None:
        qda = QuadraticDiscriminantAnalysis()
    else:
        qda = QuadraticDiscriminantAnalysis(reg_param = reg_param)
    qda.fit(X_train, y_train)
    return qda

def train_NB(X_train, y_train):
    NB = GaussianNB()
    NB.fit(X_train, y_train)
    return NB

def train_logreg(X_train, y_train, pen, c, duals = False, solver = 'liblinear'):
    lr = LogisticRegression(penalty = pen, dual = duals, C = c, solver = solver, max_iter = 100000000, multi_class = 'auto')
    lr.fit(X_train, y_train)
    return lr

def train_perceptron(X_train, y_train, pen, alpha):
    perceptron = Perceptron(penalty = pen, alpha = alpha)
    perceptron.fit(X_train, y_train)
    return perceptron

def evaluate_linear_SVM(X_train, y_train, X_test, y_test, loss_fxn, penalty, c, feature_labels, feature_type, num_ica_comps, just_ac=False, duals = True):
    X_train_standard, scaler = pre.standardise_data(X_train)
    lin_svm = train_linear_SVM(X_train_standard, y_train, loss_fxn, penalty, c, duals = duals)
    X_test_standard = scaler.transform(X_test)
    accuracy_test, accuracy_train = analyzer.evalulate_classifier("Linear SVM with L1 and c: {}".format(c), lin_svm, X_test_standard, y_test, 
                                                                  X_train_standard, y_train,
                                                                        feature_labels, feature_labels, num_ica_comps, just_accuracy=just_ac)
    return accuracy_test, lin_svm, accuracy_train


def evaluate_multiple_linsvms_for_comparison(X_train_array, X_test_array, y_train, y_test, feature_labels, feature_type, C, num_ica_comps, loss_fxn = 'squared_hinge', pen = 'l1'):
    num_feature_types = len(X_train_array)
    if num_feature_types != len(X_test_array):
        raise Exception('Something wrong with Feature array setup: train vs. test')
    
    test_accs = np.zeros((num_feature_types, len(C)))
    train_accs = np.zeros((num_feature_types, len(C)))
    features_used = np.zeros((num_feature_types, len(C)))
    for idx, c in enumerate(C):
        for j in np.arange(num_feature_types):
            #print("X train array:{} , y_train: {}, X_test: {}, y_test:{}".format(X_train_array[j].shape, y_train[j].shape, X_test_array[j].shape, y_test[j].shape))
            test_accs[j, idx], classifier, train_accs[j, idx] = evaluate_linear_SVM(X_train_array[j], y_train[j], X_test_array[j], y_test[j], 'squared_hinge', 'l1', c, feature_labels[j],
                     feature_type, num_ica_comps, just_ac = True) 
            useful, useless = analyzer.find_useful_features(classifier.coef_)
            features_used[j, idx] = len(useful)
    test_accs = np.round(test_accs*100, 2)
    train_accs = np.round(train_accs*100, 2)
    return train_accs, test_accs, features_used

if __name__ == '__main__':
    path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A'
    directory = path + '1'
        
    eeg_train, y_train, eeg_test, y_test = dl.load_pertinent_dataset(directory)
    
    # Run it for 3 class problems (hands and feet)
    y_train, eeg_train = pre.extract_3_class(y_train, eeg_train)
    y_test, eeg_test = pre.extract_3_class(y_test, eeg_test)
    
    ica_test = pre.ica(directory, eeg_test)
    ica_train = pre.ica(directory, eeg_train)
    '''
    grid, results_df = grid_search_cv_linsvm(y_train, ica_train, folds = 3, repeats = 10, is_l1 = True,
                          features = [30, 50, 100],
                              C = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], dur = 2,
                                  min_time = 4, max_time = 6)
    '''
    '''
    l1none_cv, l1none_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'l1', feature_selection_methods = [], features = [],
                           alpha = [0.0075, 0.01,  0.02, 0.03, 0.04, 0.05], dur = 2, standardise_features = True)
    
    l1ANOVA_cv, l1ANOVA_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'l1', feature_selection_methods = ['ANOVA'], features = np.arange(25, 65, 5),
                           alpha = [0.0075, 0.01,  0.02, 0.03, 0.04, 0.05], dur = 2, standardise_features = True)
    
    l1MI_cv, l1MI_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'l1', feature_selection_methods = ['MI'], features = np.arange(25, 65, 5),
                           alpha = [0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05], dur = 2, standardise_features = True)
    
    l2none_cv, l2none_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'l2', feature_selection_methods = [], features = [],
                           alpha = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005], dur = 2, standardise_features = True)
    
    l2ANOVA_cv, l2ANOVA_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'l2', feature_selection_methods = ['ANOVA'], features = np.arange(25, 65, 5),
                           alpha = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005], dur = 2, standardise_features = True)
    
    l2MI_cv, l2MI_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'l2', feature_selection_methods = ['MI'], features = [],
                           alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100], dur = 2, standardise_features = True)
    
    elasticnone_cv, elasticnone_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'elasticnet', feature_selection_methods = [], features = [],
                           alpha = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005], dur = 2, standardise_features = True)
    
    elasticANOVA_cv, elasticANOVA_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'elasticnet', feature_selection_methods = ['ANOVA'], features = np.arange(25, 65, 5),
                           alpha =  [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005], dur = 2, standardise_features = True)
    
    elasticMI_cv, elasticMI_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = 'elasticnet', feature_selection_methods = ['MI'], features = [],
                           alpha = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100], dur = 2, standardise_features = True)
    
    none_cv, none_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = None, feature_selection_methods = [], features = [],
                           alpha = [0.0001], dur = 2, standardise_features = True)
    
    noneANOVA_cv, noneANOVA_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = None, feature_selection_methods = ['ANOVA'], features = [],
                           alpha = [0.0001], dur = 2, standardise_features = True)
    
    noneMI_cv, noneMI_test, cv_acc, test_acc = repeated_k_fold_cv_perceptron(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , penalty = None, feature_selection_methods = ['MI'], features = [],
                           alpha = [0.0001], dur = 2, standardise_features = True)
   
    '''
    '''
    l1none_cv, l1none_test, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD',
                                              is_l1 = True, feature_selection_methods = [], dur = 1, C = [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1])
    l1anova_cv, l1anova_test, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD',
                                              is_l1 = True, feature_selection_methods = ['ANOVA'], features = np.arange(25, 65, 5), C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], dur = 1)
    l1mi_cv, l1mi_test, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD',
                                              is_l1 = True, feature_selection_methods = ['MI'],features = np.arange(25, 65, 5), C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], dur = 1)
    
    l2none_cv, l2none_test, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD',
                                              is_l1 = False, feature_selection_methods = [], dur = 1, C = [0.001, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1])
    l2anova_cv, l2anova_test, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD',
                                              is_l1 = False, feature_selection_methods = ['ANOVA'], features = np.arange(25, 65, 5), C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],dur = 1)
    l2mi_cv, l2mi_test, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD',
                                              is_l1 = False, feature_selection_methods = ['MI'], features = np.arange(25, 65, 5), C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],dur = 1)
    '''
    '''
    NB_cv, NB_test, cv_acc, test_acc = repeated_k_fold_cv_NB(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , feature_selection_methods = [], features = [])
    NB_cv_anova, NB_test_anova, cv_acc, test_acc = repeated_k_fold_cv_NB(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , feature_selection_methods = ['ANOVA'], features = np.arange(30, 60, 5))
    NB_cv_MI, NB_test_MI, cv_acc, test_acc = repeated_k_fold_cv_NB(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , feature_selection_methods = ['MI'], features = np.arange(30, 60, 5))
    fs_cv_yw, fs_test_yw, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'AR_Yule-Walker_PSD',
                                              is_l1 = True, feature_selection_methods = ['ANOVA', 'MI'], 
                                                  features = [])
    
    none_cv_yw, none_test_yw, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'AR_Yule-Walker_PSD',
                                              is_l1 = True, feature_selection_methods = [])
    
    fs_cv_burg, fs_test_burg, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'AR_Burg_PSD',
                                              is_l1 = True, feature_selection_methods = ['ANOVA', 'MI'], 
                                                  features = [])
  
    none_cv_burg, none_test_burg, cv_acc, test_acc = repeated_k_fold_cv_linsvm(ica_train, y_train, ica_test, y_test, folds = 3, repeats = 1, feature_extraction_method = 'AR_Burg_PSD',
                                              is_l1 = True, feature_selection_methods = [])
    lda_anova_cv, lda_anova_test, cv_acc, test_acc = repeated_k_fold_cv_DA(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_lda = True, solver = 'eigen', reg_params = ['auto'], feature_selection_methods = ['ANOVA'], features = np.arange(30, 60, 5))  
    lda_mi_cv, lda_mi_test, cv_acc, test_acc = repeated_k_fold_cv_DA(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_lda = True, solver = 'eigen', reg_params = ['auto'], feature_selection_methods = ['MI'], features = np.arange(30, 60, 5))  
    lda_none_cv, lda_none_test, cv_acc, test_acc = repeated_k_fold_cv_DA(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_lda = True, solver = 'eigen', reg_params = ['auto'], feature_selection_methods = [], features = [])  

    qda_reg_cv, qda_reg_test, cv_acc, test_acc = repeated_k_fold_cv_DA(ica_train, y_train, ica_test, y_test, folds = 5, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_lda = False, reg_params = np.arange(0, 1.1, 0.1), feature_selection_methods = ['MI'], features = [])
    
'''
'''
    l2_liblinear_none_cv, l2_liblinear_none_test, cv_acc, test_acc = repeated_k_fold_cv_logreg(ica_train, y_train, ica_test, y_test, duals = False, solver = 'liblinear', folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = False, feature_selection_methods = ['MI'], features = np.arange(25, 65, 5), C = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
                           )
    l1_liblinear_none_cv, l1_liblinear_none_test, cv_acc, test_acc = repeated_k_fold_cv_logreg(ica_train, y_train, ica_test, y_test, duals = False, solver = 'liblinear', folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = True, feature_selection_methods = ['MI'], features = np.arange(25, 65, 5), C = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
                           )
    l2_lbfgs_none_cv, l2_lbfgs_none_test, cv_acc, test_acc = repeated_k_fold_cv_logreg(ica_train, y_train, ica_test, y_test, duals = False, solver = 'lbfgs', folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = False, feature_selection_methods = ['MI'], features = np.arange(25, 65, 5), C = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
                           )
    l2_new_none_cv, l2_new_none_test, cv_acc, test_acc = repeated_k_fold_cv_logreg(ica_train, y_train, ica_test, y_test, duals = False, solver = 'newton-cg', folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = False, feature_selection_methods = ['MI'], features = np.arange(25, 65, 5), C = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
                           )
    l2_sag_none_cv, l2_sag_test, cv_acc, test_acc = repeated_k_fold_cv_logreg(ica_train, y_train, ica_test, y_test, duals = False, solver = 'sag', folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = False, feature_selection_methods = ['MI'], features = np.arange(25, 65, 5), C = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
                           )
    l2_saga_none_cv, l2_saga_test, cv_acc, test_acc = repeated_k_fold_cv_logreg(ica_train, y_train, ica_test, y_test, duals = False, solver = 'saga', folds = 3, repeats = 1, feature_extraction_method = 'Periodogram_PSD'
                       , is_l1 = False, feature_selection_methods = ['MI'], features = np.arange(25, 65, 5), C = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
                           )
'''
        
'''
    method = 'Periodogram_PSD'
    extra_args = {}
    if method == 'Periodogram_PSD':
        extra_args['window'] = 'boxcar'
    s
    #X_train, y_train, freqs = fe.extract_cwt_morlet_features(y_train, ica_train)
    #X_test, y_test, freqs = fe.extract_cwt_morlet_features(y_test, ica_test)
        
    #test_accs = np.zeros((len(C)))
    #for idx, c in enumerate(C):
     #   test_accs[idx], classifier, tmp = evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c, freqs,
      #               'TFR', 22, just_ac = False) 
      #  useful, useless = analyzer.find_useful_features(classifier.coef_)
    
    X_train, y_train, freqs = fe.extract_psd_features(y_train, ica_train, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = 0.5, frequency_precision = 1, compute_multiple_segs_per_trial = True)
    X_test, y_test, freqs = fe.extract_psd_features(y_test, ica_test, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = 0.5, frequency_precision = 1, compute_multiple_segs_per_trial = True)
    
    # 1) Linear SVM with 'l1' penalty
    
    # a) Grid search to find optimal c value
    scores_10, c_10 = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', False, "L1 SVM", folds=10)
    scores_5, c_5 = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', False, "L1 SVM", folds=5)
    scores_20, c_20 = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', False, "L1 SVM", folds=20)
    train_accs, scores_actual, features = evaluate_multiple_linsvms_for_comparison([X_train], [X_test], [y_train], [y_test], freqs, method, C, 22)
    c_actual = C[scores_actual.argmax()]
    scores_array = [np.round(scores_5*100, 2), np.round(scores_10*100, 2), np.round(scores_20*100, 2), scores_actual[0]]
    analyze_feature_performance('C', C, scores_array, train_accs, features, ['5 Fold CV', '10 Fold CV', '20 Fold CV', 'Actual'], 
                                'A01: 0.5 Second Linear l1 SVM Classification using Periodogram (Boxcar) PSD Features', 1000, metrics_computed = ['test'])
    
    print("5 Fold CV, C: {}".format(c_5))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_5['C'], freqs, method, 22, just_ac=False)
    print("10 Fold CV, C: {}".format(c_10))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_10['C'], freqs, method, 22, just_ac=False)
    print("20 Fold CV, C: {}".format(c_20))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_20['C'], freqs, method, 22, just_ac=False)
    print("Actual, C: {}".format(c_actual))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_actual, freqs, method, 22, just_ac=False)

     # b) Evaluate various classifiers trained on different amount of features using CV
    
   #for idx, c in enumerate(C):
    #    values[0, idx], classifier, ac = evaluate_linear_SVM(X_train_red2, y_train, X_test_red2, y_test, 'squared_hinge', 'l1', c, freqs_red2, ica_trainfilt.shape[0],just_ac = True) 
        #values[1, idx], classifier, ac = evaluate_linear_SVM(X_train_red, y_train, X_test_red, y_test, 'squared_hinge', 'l1', c, freqs_red, ica_trainfilt.shape[0], just_ac = True) 
        #values[2, idx], classifier,ac  = evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c, freqs, ica_trainfilt.shape[0], just_ac =True) 
    
 

    # 2) Use best classifier (reduced features) and reduce features by using filtering by rank and semi-manual component analysis 
    
    # 3) RBF Kernel SVM with reduced features
    '''