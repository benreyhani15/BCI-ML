import numpy as np
from scipy.signal import periodogram, get_window, welch
import preprocessing as pre
import data_loader as dl
from spectrum import aryule, pyule

def extract_features(data, method, extra_args, segments_per_trial, min_time = 4, max_time = 6, sampling_freq = 250, window_duration = 2):
    features_array = []
    delta = (int)(window_duration*sampling_freq)
    for i in np.arange(data.shape[2]):
        #print("\n\ncomputing features for data sample: {}".format(i)) 
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
                
                # ---- Periodogram needs extra_arg = {"fft_window"; "fft_length"} ---------
                if method == 'Periodogram_PSD':
                    fft_window = extra_args["window"]
                    fft_length = extra_args["fft_length"]
                    # beta is encoded as so: fft_window = 'kaiser (beta)'
                    if 'kaiser' in fft_window:
                        #print("kaiser window")
                        beta = (int) (fft_window[fft_window.find("(")+1:fft_window.find(")")])
                        fft_window = get_window(('kaiser', beta), datum.shape[0])
                    f, p = periodogram(datum, fs=sampling_freq, window=fft_window, nfft=fft_length, detrend=False)
                    idx = np.argwhere((f>=2) & (f<=40.25))[:,0]
                    features = p[idx]
                    feature_labels = f[idx]
                # ---- Welch needs extra_arg = {"fft_window" ; "fft_length" ; "n_per_window"} ---------
                elif method == 'Welch_PSD':
                    fft_window = extra_args["window"]
                    n_per_window = extra_args["n_per_window"]
                    fft_length = extra_args["fft_length"]
                    if 'kaiser' in fft_window:
                        #print("kaiser window")
                        # beta is encoded as so: fft_window = 'kaiser (beta)'
                        beta = (int) (fft_window[fft_window.find("(")+1:fft_window.find(")")])
                        fft_window = get_window(('kaiser', beta), n_per_window)
                    f, p = welch(datum, fs=sampling_freq, window=fft_window, nperseg = n_per_window, nfft=fft_length, detrend=False)
                    idx = np.argwhere((f>=2) & (f<=40.25))[:,0]
                    features = p[idx]  
                    feature_labels = f[idx]
                    # ---- AR Coeff: Yule-Walker needs extra_arg = {"AR_model_order"} ---------
                elif method == 'AR_Yule-Walker_Coeffs':
                    model_order = extra_args["AR_model_order"]
                    ar_coeffs, var, reflec = aryule(datum, model_order)
                    features = ar_coeffs
                    feature_labels = np.arange(1, model_order+1)
                    # ---- AR PSD: Yule-Walker needs extra_arg = {"AR_model_order" ; "fft_length"} ---------
                elif method == 'AR_Yule-Walker_PSD':
                    model_order = extra_args["AR_model_order"]
                    fft_length = extra_args["fft_length"]
                    p = pyule(datum, model_order, NFFT = fft_length, sampling = sampling_freq)
                    f = p.frequencies()
                    Pxx = p.psd
                    idx = np.argwhere((f>=2) & (f<=40.25))[:,0]
                    features = Pxx[idx] 
                    feature_labels = f[idx]
                    #      print("# of extracted features: {}".format(len(idx)))
                    
                tmp_array.append(features)
            features_array.append(tmp_array)
            start_idx = end_idx
    psd_features = np.asarray(features_array)
    #print("Shape of PSD features: {}\n".format(psd_features.shape))
    psd_features = psd_features.reshape(psd_features.shape[0], 
           psd_features.shape[1]*psd_features.shape[2])
    #print("Final features shape: {}\n".format(psd_features.shape))
    return psd_features, feature_labels

def extract_psd_features(labels, ica_data, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, sampling_freq = 250, window_duration = 2, 
                         frequency_precision = 1, compute_multiple_segs_per_trial = True):
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    fft_precision = sampling_freq/fft_length
    freq_reduction = (int)(frequency_precision/fft_precision)
    #print("need to reduce frequency by: {}".format(freq_reduction))
    labels = np.repeat(labels, segments_per_trial)
    extra_args['fft_length'] = fft_length 
    X, freqs = extract_features(ica_data, method, extra_args, segments_per_trial, min_time = min_time, max_time = max_time, 
                                sampling_freq = sampling_freq, window_duration = window_duration)
    
    # Reduce features by averaging adjacent power bins
    X_red = X.reshape(-1, freq_reduction).mean(axis=1).reshape(X.shape[0],(int) (X.shape[1]/freq_reduction))
    freqs_red = freqs.reshape(-1, freq_reduction).mean(axis=1)
    return X_red, labels, freqs_red

def extract_coeff_features(labels, ica_data, method, extra_args, min_time = 4, max_time = 6, sampling_freq = 250, window_duration = 2,
                               compute_multiple_segs_per_trial = True):
    segments_per_trial = (int)((max_time-min_time)/window_duration) if compute_multiple_segs_per_trial else 1
    labels = np.repeat(labels, segments_per_trial)
    X, coeff_labels = extract_features(ica_data, method, extra_args, segments_per_trial, min_time = min_time, max_time = max_time, 
                                sampling_freq = sampling_freq, window_duration = window_duration)
    return X, labels, coeff_labels

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