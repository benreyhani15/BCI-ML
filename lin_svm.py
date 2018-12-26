import preprocessing as pre
import feature_extraction as fe
import classifier
import data_loader as dl
import numpy as np

def find_optimal_c_cv(X_train, y_train, pen, loss_fxn, send_notif, title, folds=10):
    param_grid_linsvm = {'C': [0.01, 0.0125, 0.015, 0.0175, 0.1, 1, 10, 100, 1000]}
    grid_search_linsvm = classifier.find_linear_SVM(X_train, y_train, param_grid_linsvm, pen, loss_fxn, send_notif,title, folds)
    return grid_search_linsvm.cv_results_['mean_test_score']

def evaluate_linSVM_l1(X_train, y_train, X_test, y_test, c, freqs, num_components):
     return classifier.evaluate_linear_SVM(X_train, y_train, X_test, y_test,"squared_hinge", 'l1', c, freqs, num_components)
    
def run_it():
    path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A1'
    eeg_train, eeg_test, eeg_trainfil, eeg_testfil, y_train, y_test = dl.load_dataset(path)
    
    #ica_train = pre.ica(path, eeg_train, True, False)
    ica_test = pre.ica(path, eeg_test, False, False)
    ica_trainfil = pre.ica(path, eeg_trainfil, True, True)
    #ica_test_filt = pre.ica(path, eeg_testfil, False, True)
    
    # Train set can be filtered from continous data, Test set when in epochs
    #X_train, freqs = fe.compute_psdp(ica_train, fft_length = 1024, fft_window = 'boxcar')
    X_test, freqs1 = fe.compute_psdp(ica_test, fft_length = 1024, fft_window = 'boxcar')
    X_train, freqs = fe.compute_psdp(ica_trainfil, fft_length = 1024, fft_window = 'boxcar')
    #X_testfilt, freqs4 = fe.compute_psdp(ica_test_filt, fft_length = 1024, fft_window = 'boxcar')
    

    #Linear SVM with l1 penalty:
    #cv_score = find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter")
    #cv_scoreF = find_optimal_c_cv(X_trainfilt, y_train, 'l1', 'squared_hinge', True, "Filter")
    weights = evaluate_linSVM_l1(X_train, y_train, X_test, y_test, 0.015,freqs, ica_trainfil.shape[0]) 

if __name__ == '__main__':
    run_it()