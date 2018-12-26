import data_loader as dl
import preprocessing as pre
import lin_svm
import feature_extraction as fe
import numpy as np
import scipy as sp

def run_ICA_filt_and_CV_test():
    path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A1'
    eeg_train, eeg_test, eeg_trainfil, eeg_testfil, y_train, y_test = dl.load_dataset(path)
    
    ica_train = pre.ica(path, eeg_train, True, False)
    ica_test = pre.ica(path, eeg_test, False, False)
    ica_trainfilt = pre.ica(path, eeg_trainfil, True, True)
    ica_test_filt = pre.ica(path, eeg_testfil, False, False)
    
    X_train, freqs = fe.compute_psdp(ica_train, fft_length = 1024, fft_window = 'boxcar')
    X_test, freqs2 = fe.compute_psdp(ica_test, fft_length = 1024, fft_window = 'boxcar')
    X_trainfilt, freqs3 = fe.compute_psdp(ica_trainfilt, fft_length = 1024, fft_window = 'boxcar')
    X_testfilt, freqs4 = fe.compute_psdp(ica_test_filt, fft_length = 1024, fft_window = 'boxcar')
    
    values = np.zeros((6,len(C_array)))
    values[0] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', False, "No Filter")
    values[1] = lin_svm.find_optimal_c_cv(X_trainfilt, y_train, 'l1', 'squared_hinge', False, "Filter")
    C_array = np.arange(0.0025, 0.09,0.0025)
    for idx, c in enumerate(C_array):
        values[2,idx] = lin_svm.evaluate_linSVM_l1(X_train, y_train, X_test, y_test, c,freqs, ica_train.shape[0]) 
        values[3, idx] = lin_svm.evaluate_linSVM_l1(X_trainfilt, y_train, X_test, y_test, c,freqs, ica_train.shape[0]) 
        values[4,idx] = lin_svm.evaluate_linSVM_l1(X_train, y_train, X_testfilt, y_test, c,freqs, ica_train.shape[0]) 
        values[5, idx] = lin_svm.evaluate_linSVM_l1(X_trainfilt, y_train, X_testfilt, y_test, c,freqs, ica_train.shape[0]) 
    get_experiment_results(values)
    return values

def evaluate_different_ica_components():
    path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A1'
    eeg_train, eeg_test, eeg_trainfil, eeg_testfil, y_train, y_test = dl.load_dataset(path)
    
    ica_train = pre.ica(path, eeg_trainfil, False, True)
    ica_test = pre.ica(path, eeg_test, False, False)
    
    X_train, freqs = fe.compute_psdp(ica_train, fft_length = 1024, fft_window = 'boxcar')
    X_test, freqs2 = fe.compute_psdp(ica_test, fft_length = 1024, fft_window = 'boxcar')
    
    for C in [0.001, 0.01, 0.0125, 0.015, 0.0175, 0.1, 1, 10, 100, 1000]:
        lin_svm.evaluate_linSVM_l1(X_train, y_train, X_test, y_test,C ,freqs, ica_train.shape[0]) 
      
def test_CV_folds():
    path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A1'
    eeg_train, eeg_test, eeg_trainfil, eeg_testfil, y_train, y_test = dl.load_dataset(path)
    
    ica_train = pre.ica(path, eeg_trainfil, False, True)
    ica_test = pre.ica(path, eeg_test, False, False)
    
    X_train, freqs = fe.compute_psdp(ica_train, fft_length = 1024, fft_window = 'boxcar')
    X_test, freqs2 = fe.compute_psdp(ica_test, fft_length = 1024, fft_window = 'boxcar')
    
    C = [0.01, 0.0125, 0.015, 0.0175, 0.1, 1, 10, 100, 1000]
    
    values = np.zeros((9, len(C)))

    values[0] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=5)
    values[1] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=10)
    values[2] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=20)
    values[3] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=30)
    values[4] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=40)
    values[5] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=50)
    values[6] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=60)
    values[7] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds=70)
    values[8] = lin_svm.find_optimal_c_cv(X_train, y_train, 'l1', 'squared_hinge', True, "No Filter", folds = ((int)(len(y_train)/4)))

    actual = np.zeros(len(C))
    for idx, c in enumerate(C):
        actual[idx] = lin_svm.evaluate_linSVM_l1(X_train, y_train, X_test, y_test, c ,freqs, ica_train.shape[0]) 
    
    values = values-actual
    return (values*100)

def get_experiment_results(values):
    diff = np.abs(values[0] - values[1])
    print("Diff of CV score stats: mean:{}, var:{}, max:{}, min:{}\n".format(diff.mean(), diff.var(), diff.max(), diff.min()))
    results = values[2:6]
    avg_var = results.var(axis=0).mean()
    print("Results average variance was: {}\n".format(avg_var))
    diff_combs = np.zeros((results.shape[1], int(sp.special.comb(results.shape[0], 2))))
    for i in np.arange(results.shape[1]):
        result = np.zeros((results.shape[0], 1))
        result[:,0] = results[:, i]
        diff_combs[i, :] = np.abs(sp.spatial.distance.pdist(result, 'cityblock')) 
    print("Diff of combination of filtering and no filtering score stats: mean:{}, var:{}, max:{}, min:{}\n".format
          (diff_combs.mean(), diff_combs.var(), diff_combs.max(), diff_combs.min()))
    cv_avg_estimate = values[0:2].mean(axis=0)
    result_and_cv_diff = results - cv_avg_estimate
    print("Statistics of the difference between actual accuracy and accuracy predicted by CV: mean:{}, var:{}, max:{}, min:{}\n".
          format(result_and_cv_diff.mean(), result_and_cv_diff.var(), result_and_cv_diff.max(), result_and_cv_diff.min()))

    
        
if __name__ == '__main__':
    run_ICA_filt_and_CV_test()