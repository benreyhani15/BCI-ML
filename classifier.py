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

path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A'

def kfold_cv(classifier, X_train, y_train, param_grid, send_notif, title, folds=10):
    grid_search = GridSearchCV(classifier, param_grid, cv=folds)
    grid_search.fit(X_train, y_train)
    analyzer.display_score_matrix(grid_search)
    if send_notif: 
        send_email_notification("{}\n\nResults for search: {}".format(title, analyzer.get_string_results(grid_search)))
    return grid_search

def find_linear_SVM(X_train, y_train, param_grid, pen, loss_fxn, send_notif, title, folds=10):
    X_train, scaler = pre.standardise_data(X_train)
    duals = True 
    if pen == 'l1':
        duals = False
    lin_svm = svm.LinearSVC(penalty=pen, loss=loss_fxn, dual=duals, max_iter = 1000000)
    print(lin_svm)
    grid_search = kfold_cv(lin_svm, X_train, y_train, param_grid, send_notif, title, folds)
    return grid_search.cv_results_['mean_test_score'], grid_search.best_params_
    
def train_linear_SVM(X_train, y_train, loss_fxn, pen, c):
    duals = True
    if pen == 'l1':
        duals = False
    lin_svm = svm.LinearSVC(penalty = pen, loss = loss_fxn, dual = duals, C = c)
    lin_svm.fit(X_train, y_train)
    return lin_svm
    
def evaluate_linear_SVM(X_train, y_train, X_test, y_test, loss_fxn, penalty, c, feature_labels, feature_type, num_ica_comps, just_ac=False):
    X_train_standard, scaler = pre.standardise_data(X_train)
    lin_svm = train_linear_SVM(X_train_standard, y_train, loss_fxn, penalty, c)
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
    
    C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    #C = np.linspace(0.001, 0.1, 100)
    param_grid_linsvm = {'C': C}
    
    method = 'Periodogram_PSD'
    extra_args = {}
    if method == 'Periodogram_PSD':
        extra_args['window'] = 'boxcar'
        
    X_train, y_train, freqs = fe.extract_psd_features(y_train, ica_train, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = 0.5, frequency_precision = 1, compute_multiple_segs_per_trial = False)
    X_test, y_test, freqs = fe.extract_psd_features(y_test, ica_test, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = 0.5, frequency_precision = 1, compute_multiple_segs_per_trial = False)
    
    # 1) Linear SVM with 'l1' penalty
    
    # a) Grid search to find optimal c value
    scores_10, c_10 = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', False, "L1 SVM", folds=10)
    scores_5, c_5 = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', False, "L1 SVM", folds=5)
    scores_20, c_20 = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', False, "L1 SVM", folds=20)
    scores_40, c_40 = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', False, "L1 SVM", folds=40)
    train_accs, scores_actual, features = evaluate_multiple_linsvms_for_comparison([X_train], [X_test], [y_train], [y_test], freqs, method, C, 22)
    c_actual = C[scores_actual.argmax()]
    scores_array = [np.round(scores_5*100, 2), np.round(scores_10*100, 2), np.round(scores_20*100, 2), np.round(scores_40*100, 2), scores_actual[0]]
    analyze_feature_performance('C', C, scores_array, train_accs, features, ['5 Fold CV', '10 Fold CV', '20 Fold CV', '40 Fold CV', 'Actual'], 
                                'A01: 0.5 Second Linear l1 SVM Classification using Periodogram (Boxcar) PSD Features', 1000, metrics_computed = ['test'])
    '''
    print("5 Fold CV, C: {}".format(c_5))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_5['C'], freqs, method, 22, just_ac=False)
    print("10 Fold CV, C: {}".format(c_10))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_10['C'], freqs, method, 22, just_ac=False)
    print("20 Fold CV, C: {}".format(c_20))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_20['C'], freqs, method, 22, just_ac=False)
    print("40 Fold CV, C: {}".format(c_40))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_40['C'], freqs, method, 22, just_ac=False)
    print("Actual, C: {}".format(c_actual))
    evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c_actual, freqs, method, 22, just_ac=False)
    '''
     # b) Evaluate various classifiers trained on different amount of features using CV
    
   #for idx, c in enumerate(C):
    #    values[0, idx], classifier, ac = evaluate_linear_SVM(X_train_red2, y_train, X_test_red2, y_test, 'squared_hinge', 'l1', c, freqs_red2, ica_trainfilt.shape[0],just_ac = True) 
        #values[1, idx], classifier, ac = evaluate_linear_SVM(X_train_red, y_train, X_test_red, y_test, 'squared_hinge', 'l1', c, freqs_red, ica_trainfilt.shape[0], just_ac = True) 
        #values[2, idx], classifier,ac  = evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c, freqs, ica_trainfilt.shape[0], just_ac =True) 
    
 

    # 2) Use best classifier (reduced features) and reduce features by using filtering by rank and semi-manual component analysis 
    
    # 3) RBF Kernel SVM with reduced features