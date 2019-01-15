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
    lin_svm = svm.LinearSVC(penalty=pen, loss=loss_fxn, dual=duals, max_iter = 100000000000)
    print(lin_svm)
    grid_search = kfold_cv(lin_svm, X_train, y_train, param_grid, send_notif, title, folds)
    return grid_search.cv_results_['mean_test_score']
    
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
    directory = path + '1'
    
    rejected_idx, eeg_train, eeg_test, eeg_trainfil, eeg_testfil, y_train, y_test = dl.load_dataset(directory)
    
    # Run it for 3 class problems (hands and feet)
    train_idx = np.where(y_train != 4)[0]
    test_idx = np.where(y_test != 4)[0]
  
    y_train = y_train[np.where(y_train != 4)]
    y_test = y_test[np.where(y_test != 4)]
    
    eeg_trainfil = eeg_trainfil[:, :, train_idx]
    eeg_test = eeg_test[:, :, test_idx]
    
    ica_test = pre.ica(directory, eeg_test)
    ica_trainfilt = pre.ica(directory, eeg_trainfil)    
    
    window_length = 0.25
    min_time = 4
    max_time = 6
    segments_per_trial = (int)((max_time-min_time)/window_length)
   
    y_train = np.repeat(y_train, segments_per_trial)
    y_test = np.repeat(y_test, segments_per_trial)
    X_test, freqs = fe.compute_psdp(ica_test, fft_length = 1024, fft_window = 'boxcar', window_duration = window_length)
    X_train, freqs = fe.compute_psdp(ica_trainfilt, fft_length = 1024, fft_window = 'boxcar', window_duration = window_length)
    
    #C = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    C = np.arange(0.001, 0.006, 0.00025)
    #C = np.linspace(0.001, 0.05)
    param_grid_linsvm = {'C': C}
    
    values = np.zeros((3, len(C)))
       
    X_train_red =  X_train.reshape(-1, 2).mean(axis=1).reshape(X_train.shape[0],(int) (X_train.shape[1]/2))
    X_test_red =  X_test.reshape(-1, 2).mean(axis=1).reshape(X_test.shape[0],(int) (X_test.shape[1]/2))
    freqs_red = freqs.reshape(-1, 2).mean(axis=1)

    X_train_red2 = X_train_red.reshape(-1, 2).mean(axis=1).reshape(X_train_red.shape[0],(int) (X_train_red.shape[1]/2))
    X_test_red2 = X_test_red.reshape(-1, 2).mean(axis=1).reshape(X_test_red.shape[0],(int) (X_test_red.shape[1]/2))
    freqs_red2 = freqs_red.reshape(-1, 2).mean(axis=1)
    
    # 1) Linear SVM with 'l1' penalty
    
    # a) Grid search to find optimal c value
    #scores_full = find_linear_SVM(X_train, y_train, param_grid_linsvm, 'l1', 'squared_hinge', True, "Full PSDs L1 SVM", folds=10)
    #scores_red = find_linear_SVM(X_train_red, y_train, param_grid_linsvm, 'l1', 'squared_hinge', True, "Red PSDs L1 SVM", folds=10)
    #scores_red2 = find_linear_SVM(X_train_red2, y_train, param_grid_linsvm, 'l1', 'squared_hinge', True, "Red2 PSDs L1 SVM", folds=10)
    #scores_3class = find_linear_SVM(X_train_3class, y_train_3class, param_grid_linsvm, 'l1', 'squared_hinge', True, "3class L1 SVM", folds=10)
    
     # b) Evaluate various classifiers trained on different amount of features using CV
    
    for idx, c in enumerate(C):
        values[0, idx], classifier, ac = evaluate_linear_SVM(X_train_red2, y_train, X_test_red2, y_test, 'squared_hinge', 'l1', c, freqs_red2, ica_trainfilt.shape[0],just_ac = True) 
        #values[1, idx], classifier, ac = evaluate_linear_SVM(X_train_red, y_train, X_test_red, y_test, 'squared_hinge', 'l1', c, freqs_red, ica_trainfilt.shape[0], just_ac = True) 
        #values[2, idx], classifier,ac  = evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', c, freqs, ica_trainfilt.shape[0], just_ac =True) 
    
    # d) Evaluate and obtain metrics on evaluation set
    accuracy_test, lin_svm, ac = evaluate_linear_SVM(X_train_red2, y_train, X_test_red2, y_test, 'squared_hinge', 'l1', 0.004, freqs_red2, ica_trainfilt.shape[0]) 
    #accuracy_red, lin_svmred, ac = evaluate_linear_SVM(X_train_red, y_train, X_test_red, y_test, 'squared_hinge', 'l1', 0.0175, freqs_red, ica_trainfilt.shape[0])
    #accuracy, lin_svm, ac = evaluate_linear_SVM(X_train, y_train, X_test, y_test, 'squared_hinge', 'l1', 0.015, freqs, ica_trainfilt.shape[0])

    analyzer.find_useful_features(lin_svm.coef_)
    #analyzer.find_useful_features(lin_svmred.coef_)
    #analyzer.find_useful_features(lin_svm.coef_)
    

    # c) Plot and see difference between amount/preciseness of PSD features used 
    #print("a-b stats: mean={}, var={}".format(np.mean(values[0]-values[1]), np.var(values[0]-values[1])))
    #print("a-c stats: mean={}, var={}".format(np.mean(values[0]-values[2]), np.var(values[0]-values[2])))
    #print("b-c stats: mean={}, var={}".format(np.mean(values[1]-values[2]), np.var(values[1]-values[2])))
    df=pd.DataFrame({'C': C, 'a': values[0], 'b': values[1], 'c':values[2]})
    plt.plot( 'C', 'a', data=df, marker='o', markerfacecolor='blue', markersize=12, linewidth=4)
    #plt.plot( 'C', 'b', data=df, marker='o', markerfacecolor='red', markersize=12, linewidth=4)
    #plt.plot( 'C', 'c', data=df, marker='o', markerfacecolor='black', markersize=12, linewidth=4)
    plt.xlabel('C')
    plt.ylabel('% Accuracy')
    plt.legend()
    plt.show()
    # 2) Use best classifier (reduced features) and reduce features by using filtering by rank and semi-manual component analysis 
    
    # 3) RBF Kernel SVM with reduced features