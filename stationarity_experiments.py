import preprocessing as pre
import data_loader as dl
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
    
def run_stationarity_tests(X, y, window_duration, method, min_time = 4, max_time = 6, sampling_freq = 250, compute_multiple_segs_per_trial = True):
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    test_stat_array = np.zeros(X.shape[0])
    p_array = np.zeros(X.shape[0])
    delta = (int)(window_duration*sampling_freq)
    for i in np.arange(X.shape[2]):
        #print("\n\ncomputing values for data sample: {}".format(i)) 
        start_idx = (int)(sampling_freq * min_time)
        for k in np.arange(segments_per_trial):
            #print("For data segment: {}".format(k))
            for j in np.arange(X.shape[0]):
                #print("Working on component: {}".format(j))
                end_idx = start_idx+delta
                #print("start: {}, end:{}".format(start_idx, end_idx))
                datum = X[j, start_idx :end_idx, i]
                tau = 0
                p = 0
                if method == 'adfuller':
                    tau, p = adfuller(datum[:, 0])[0:2]
                elif method == 'kpss':
                    tau, p = kpss(datum[:, 0])[0:2]
                test_stat_array[j]+=tau
                p_array[j]+=p
    return test_stat_array.mean(), p_array.mean()    
            
def obtain_stationarity_test_results_per_class(X, y, window_duration, method, min_time = 4, max_time = 6, sampling_freq = 250, get_multiple_segs_per_trial = True):
    if method != 'adfuller' and method != 'kpss': raise Exception("Check method passed into function")
    labels = np.unique(y)
    test_stat_array = np.zeros((len(labels), X.shape[0]))
    p_array = np.zeros((len(labels), X.shape[0]))
    for value, label in enumerate(labels):
        idx = np.argwhere(y == label)
        X_tmp = X[:, :, idx]
        y_tmp = y[idx]
        test_stat_array[value], p_array[value] = run_stationarity_tests(X_tmp, y_tmp, window_duration, method, min_time = min_time, max_time = max_time, 
                       sampling_freq = sampling_freq, compute_multiple_segs_per_trial = get_multiple_segs_per_trial)
    return test_stat_array, p_array
    
if __name__ == '__main__':
    path = '/Users/benreyhani/Files/GradSchool/BCISoftware/main/BCI/Dataset/A'
    directory = path + '1'
        
    eeg_train, y_train, eeg_test, y_test = dl.load_pertinent_dataset(directory)
    
    # Run it for 3 class problems (hands and feet)
    y_train, eeg_train = pre.extract_3_class(y_train, eeg_train)
    y_test, eeg_test = pre.extract_3_class(y_test, eeg_test)
    
    ica_test = pre.ica(directory, eeg_test)
    ica_train = pre.ica(directory, eeg_train)
    
    eeg = np.concatenate((eeg_train, eeg_test), axis=2)
    ica = np.concatenate((ica_train, ica_test), axis=2)
    y = np.concatenate((y_train, y_test), axis=0)
    
    eeg_stats_adfuller, eeg_p_adfuller = obtain_stationarity_test_results_per_class(eeg, y, 2, 'adfuller')
    ica_stats_adfuller, ica_p_adfuller = obtain_stationarity_test_results_per_class(ica, y, 2, 'adfuller')
    eeg_stats_kpss, eeg_p_kpss = obtain_stationarity_test_results_per_class(eeg, y, 2, 'kpss')
    ica_stats_kpss, ica_p_kpss = obtain_stationarity_test_results_per_class(ica, y, 2, 'kpss')

    # TODO: Get rid of rejected trials for analysis ? 
    
    