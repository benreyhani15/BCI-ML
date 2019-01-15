import numpy as np
from scipy.signal import periodogram, get_window, welch
import preprocessing as pre
import data_loader as dl
import pandas as pd
from spectrum import aryule

def compute_psdp(data, fft_length = 1024, fft_window = "boxcar", min_time = 4, 
                 max_time = 6, sampling_freq = 250, window_duration = 2, compute_multiple_segs_per_trial = True):
    features_array = []
    delta = (int)(window_duration*sampling_freq)
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    for i in np.arange(data.shape[2]):
        #print("\n\ncomputing PSDs for data sample: {}".format(i)) 
        start_idx = (int)(sampling_freq * min_time)
        for k in np.arange(segments_per_trial):
            #print("For data segment: {}".format(k))
            tmp_array = []
            for j in np.arange(data.shape[0]):
                #print("Working on component: {}".format(j))
                end_idx = start_idx+delta
                #print("start: {}, end:{}".format(start_idx, end_idx))
                datum = data[j, start_idx :end_idx, i]
                #print("Shape of datum: {}".format(datum.shape))
                # beta is encoded as so: fft_window = 'kaiser (beta)'
                if 'kaiser' in fft_window:
                   #print("kaiser window")
                    beta = (int) (fft_window[fft_window.find("(")+1:fft_window.find(")")])
                    fft_window = get_window(('kaiser', beta), datum.shape[0])
                f, p = periodogram(datum, fs=sampling_freq, window=fft_window, nfft=fft_length, detrend=False)
                idx = np.argwhere((f>=2) & (f<=40.25))[:,0]
                psd_features = p[idx]
         #      print("# of extracted features: {}".format(len(idx)))
                tmp_array.append(psd_features)
            features_array.append(tmp_array)
            start_idx = end_idx
    psd_features = np.asarray(features_array)
    #print("Shape of PSD features: {}\n".format(psd_features.shape))
    psd_features = psd_features.reshape(psd_features.shape[0], 
           psd_features.shape[1]*psd_features.shape[2])
    #print("Final features shape: {}\n".format(psd_features.shape))
    return psd_features, f[idx]

def compute_psd_welch(data, n_per_window, fft_length = 1024, fft_window = "boxcar", min_time = 4, 
                 max_time = 6, sampling_freq = 250, window_duration = 2, compute_multiple_segs_per_trial = True):
    # Using 50% Overlap between different FFT windows
    features_array = []
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    delta = (int)(window_duration*sampling_freq)
    for i in np.arange(data.shape[2]):
        #print("\n\ncomputing PSDs for data sample: {}".format(i)) 
        start_idx = (int)(sampling_freq * min_time)
        for k in np.arange(segments_per_trial):
            #print("For data segment: {}".format(k))
            tmp_array = []
            for j in np.arange(data.shape[0]):
                #print("Working on component: {}".format(j))
                end_idx = start_idx+delta
                #print("start: {}, end:{}".format(start_idx, end_idx))
                datum = data[j, start_idx :end_idx, i]
                #print("Shape of datum: {}".format(datum.shape))
                if 'kaiser' in fft_window:
                   #print("kaiser window")
                   # beta is encoded as so: fft_window = 'kaiser (beta)'
                    beta = (int) (fft_window[fft_window.find("(")+1:fft_window.find(")")])
                    fft_window = get_window(('kaiser', beta), n_per_window)
                f, p = welch(datum, fs=sampling_freq, window=fft_window, nperseg = n_per_window, nfft=fft_length, detrend=False)
                idx = np.argwhere((f>=2) & (f<=40.25))[:,0]
                psd_features = p[idx]
         #      print("# of extracted features: {}".format(len(idx)))
                tmp_array.append(psd_features)
            features_array.append(tmp_array)
            start_idx = end_idx
    psd_features = np.asarray(features_array)
    #print("Shape of PSD features: {}\n".format(psd_features.shape))
    psd_features = psd_features.reshape(psd_features.shape[0], 
           psd_features.shape[1]*psd_features.shape[2])
    #print("Final features shape: {}\n".format(psd_features.shape))
    return psd_features, f[idx]

def extract_AR_YW_coeff_features(data, model_order, min_time = 4, max_time = 6, sampling_freq = 250,
                                 window_duration = 2, compute_multiple_segs_per_trial = True):
    features_array = []
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    delta = (int)(window_duration*sampling_freq)
    for i in np.arange(data.shape[2]):
        #print("\n\ncomputing PSDs for data sample: {}".format(i)) 
        start_idx = (int)(sampling_freq * min_time)
        for k in np.arange(segments_per_trial):
            #print("For data segment: {}".format(k))
            tmp_array = []
            for j in np.arange(data.shape[0]):
                #print("Working on component: {}".format(j))
                end_idx = start_idx+delta
                #print("start: {}, end:{}".format(start_idx, end_idx))
                datum = data[j, start_idx :end_idx, i]
                #print("Shape of datum: {}".format(datum.shape))              
                ar_coeffs, var, reflec_coeffs = aryule(datum, model_order)
         #      print("# of extracted features: {}".format(len(idx)))
                tmp_array.append(ar_coeffs)
            features_array.append(tmp_array)
            start_idx = end_idx
    ar_coeff_features = np.asarray(features_array)
    #print("Shape of AR Coeff features: {}\n".format(ar_coeff_features.shape))
    ar_coeff_features = ar_coeff_features.reshape(ar_coeff_features.shape[0], 
           ar_coeff_features.shape[1]*ar_coeff_features.shape[2])
    #print("Final features shape: {}\n".format(ar_coeff_features.shape))
    print("Derp: {}".format(ar_coeff_features.shape))
    return ar_coeff_features   

def extract_AR_YW_coeffs_dataset(labels, ica_data, model_order, min_time = 4, max_time = 6, sampling_freq = 250,
                                 window_duration = 2, compute_multiple_segs_per_trial = True):
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    labels = np.repeat(labels, segments_per_trial)
    X = extract_AR_YW_coeff_features(ica_data, model_order, min_time = min_time, max_time = max_time, sampling_freq = sampling_freq,
                                 window_duration = window_duration, compute_multiple_segs_per_trial = compute_multiple_segs_per_trial)
    return X, labels

def extract_psdP_features_using_windows(labels, ica_data, fft_length = 1024, fft_window = "boxcar", min_time = 4, 
                 max_time = 6, sampling_freq = 250, window_duration = 2, frequency_precision = 1, compute_multiple_segs_per_trial = True):
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    fft_precision = sampling_freq/fft_length
    freq_reduction = (int)(frequency_precision/fft_precision)
    #print("need to reduce frequency by: {}".format(freq_reduction))
    
    labels = np.repeat(labels, segments_per_trial)
    X, freqs = compute_psdp(ica_data, fft_length=fft_length, fft_window=fft_window, 
                            window_duration = window_duration, compute_multiple_segs_per_trial = compute_multiple_segs_per_trial)
    
    # Reduce features by averaging adjacent power bins
    X_red = X.reshape(-1, freq_reduction).mean(axis=1).reshape(X.shape[0],(int) (X.shape[1]/freq_reduction))
    freqs_red = freqs.reshape(-1, freq_reduction).mean(axis=1)
    return X_red, labels, freqs_red
    
def extract_psdW_features_using_windows(labels, ica_data, time_per_seg, fft_length = 1024, fft_window = "boxcar", min_time = 4, 
                 max_time = 6, sampling_freq = 250, window_duration = 2, frequency_precision = 1, compute_multiple_segs_per_trial = True):
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    fft_precision = sampling_freq/fft_length
    freq_reduction = (int)(frequency_precision/fft_precision)
    #print("need to reduce frequency by: {}".format(freq_reduction))
    
    labels = np.repeat(labels, segments_per_trial)
    
    nper_welch_seg = (int)(time_per_seg * sampling_freq)
    X, freqs = compute_psd_welch(ica_data, nper_welch_seg, fft_length=fft_length, fft_window=fft_window, 
                                 window_duration = window_duration, compute_multiple_segs_per_trial = compute_multiple_segs_per_trial)
    
    # Reduce features by averaging adjacent power bins
    X_red = X.reshape(-1, freq_reduction).mean(axis=1).reshape(X.shape[0],(int) (X.shape[1]/freq_reduction))
    freqs_red = freqs.reshape(-1, freq_reduction).mean(axis=1)
    return X_red, labels, freqs_red

if __name__ == "__main__":
    # Testing different PSD estimating techniques for short data samples
    path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A'
    directory = path + '1'
    
    rejected_trials, eeg_train, eeg_test, eeg_trainfil, eeg_testfil, y_train, y_test = dl.load_dataset(directory)
    
    # Exclude rejected trials from train set
    #y_train = np.delete(y_train, rejected_trials)
    #eeg_trainfil = np.delete(eeg_trainfil, rejected_trials, axis=2)
    
    # Run it for 3 class problems (hands and feet)
    y_train, eeg_train = pre.extract_3_class(y_train, eeg_trainfil)
    y_test, eeg_test = pre.extract_3_class(y_test, eeg_test)
    
    ica_test = pre.ica(directory, eeg_test, algorithm='extended-infomax')
    ica_train = pre.ica(directory, eeg_train, algorithm='extended-infomax')
    '''
    TEST_TYPE = 'periodogram_dp_comparison'
    if TEST_TYPE == 'periodogram_dur_comparison':
        #print("periodogram")
        periodogram_window_durations = [2, 1, 0.5, .25]
        periodogram_classification_duration_comparison(y_train, y_test, ica_train, ica_test, periodogram_window_durations, window = 'boxcar')
    elif TEST_TYPE == 'welch_dur_comparison':  
        #print("welch")
        welch_window_durations = [0.25, 0.125]
        classification_window_duration = 0.5
        welch_window_duration_comparison(y_train, y_test, ica_train, ica_test, classification_window_duration, welch_window_durations, window = 'boxcar')
    elif TEST_TYPE == 'periodogram_window_comparison':
        windows = ['boxcar', 'hamming', 'kaiser (7)', 'kaiser (10)', 'kaiser (14)']
        #periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 2, windows, freq_prec = 1)
        periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 1, windows, freq_prec = 1)
        #periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 0.5, windows, freq_prec = 1)
        #periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 0.25, windows, freq_prec = 1)
    elif TEST_TYPE == 'periodogram_dp_comparison':
        test_accs = periodogram_classification_wrt_data_points_analysis(y_train, y_test, ica_train, ica_test, window = 'boxcar', frequency_prec = 1)

        '''