import data_loader as dl
import preprocessing as pre
import plotter, classifier
import numpy as np
import pandas as pd
from feature_extraction import extract_psd_features, extract_coeff_features
import matplotlib.pyplot as plt
from analyzer import analyze_feature_performance
        
# Experiment to see the effects of classification accuracy using different time windows for classification
def periodogram_classification_duration_comparison(y_train, y_test, ica_train, ica_test, window_durations, window = 'boxcar', freq_prec = 1):
    # Doesn't support overlapping windows yet
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    freqs = []
    var_array = []
    extra_args = {"window":window}
    for idx, window_dur in enumerate(window_durations):
        X_train, y_train_tmp, freq = extract_psd_features(y_train, ica_train, 'Periodogram_PSD', extra_args , fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = window_dur, frequency_precision = freq_prec, compute_multiple_segs_per_trial = True)
        X_test, y_test_tmp, freq = extract_psd_features(y_test, ica_test, 'Periodogram_PSD', extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = window_dur, frequency_precision = freq_prec, compute_multiple_segs_per_trial = True)
        X_train_array.append(X_train)
        y_train_array.append(y_train_tmp)
        X_test_array.append(X_test)
        y_test_array.append(y_test_tmp)
        freqs.append(freq)
        var_array.append("{}s".format(window_dur))

    C = np.linspace(0.001, 0.1, 100)
    num_ica_comps = ica_train.shape[0]
    train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(np.asarray(X_train_array), np.asarray(X_test_array),
                                                        np.asarray(y_train_array), np.asarray(y_test_array), 
                                                            np.asarray(freqs), 'PSD', C, num_ica_comps, 
                                                                loss_fxn = 'squared_hinge', pen = 'l1')
    title = "Periodogram-{} Window".format(window)
    analyze_feature_performance('C', C, test_accs, train_accs, features_used, var_array, title, X_train_array[0].shape[1])

#Comparison of different model order 
def AR_model_order_comparison(y_train, y_test, ica_train, ica_test, classification_duration, method, use_psd_features = False):
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    feature_labels = []
    
    #model_orders = [16, 29,  34]
    model_orders = np.arange(23, 50, 1)
    #C = np.linspace(0.001, 0.1, 100)
    #C = [0.001, 0.005, 0.01,  0.05,  0.1, 0.5, 1, 5, 10, 30, 50, 75, 100, 200, 350, 500, 650, 800, 1000]
    C = np.linspace(0.01, 1, 300)
    #C = np.linspace(20, 1000, 500)
    #C = np.arange(2, 500)
    #C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    
    #C = np.linspace(0.0075, 0.3, 200)
    for idx, model_order in enumerate(model_orders):
        extra_args = {"AR_model_order": model_order}
        if use_psd_features:
            X_train, y_train_tmp, freqs = extract_psd_features(y_train, ica_train, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = classification_duration, frequency_precision = 1, compute_multiple_segs_per_trial = True)
            X_test, y_test_tmp, freqs = extract_psd_features(y_test, ica_test, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = classification_duration, frequency_precision = 1, compute_multiple_segs_per_trial = True)
            feature_labels.append(freqs)
        else:
            X_train, y_train_tmp, coeff_labels = extract_coeff_features(y_train, ica_train, method, extra_args, min_time = 4, max_time = 6, sampling_freq = 250, 
                                    window_duration = classification_duration, compute_multiple_segs_per_trial = True)
            X_test, y_test_tmp, coeff_labels = extract_coeff_features(y_test, ica_test, method, extra_args, min_time = 4, max_time = 6, sampling_freq = 250, 
                                    window_duration = classification_duration, compute_multiple_segs_per_trial = True)
            feature_labels.append(coeff_labels)
        X_train_array.append(X_train)
        y_train_array.append(y_train_tmp)
        X_test_array.append(X_test)
        y_test_array.append(y_test_tmp)
           
    num_ica_comps = ica_train.shape[0]
    feature_type = 'PSD' if use_psd_features else 'Coeffs'
    
    train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(X_train_array, X_test_array,
                                                        y_train_array, y_test_array, feature_labels, feature_type, C, num_ica_comps, loss_fxn = 'squared_hinge', pen = 'l1')
    
    plotter.plot_2d_annotated_heatmap(test_accs, "{}s - {} Features' Test Classification Accuracy".format(classification_duration, method), 'C', 'Model Order', C, model_orders)
    
    title =  "{}s - {}".format(classification_duration, method)
    feature_count_label = "{}".format(X_test_array[0].shape[1]) if use_psd_features else "(Model Order x 22)"
    
    analyze_feature_performance('C', C, test_accs, train_accs, features_used, np.asarray(model_orders, str), title, feature_count_label)
    #plotter.plot_2d_annotated_heatmap(train_accs, "Train Classification Accuracy", 'C', 'Model Order', C, model_orders)
    #plotter.plot_2d_annotated_heatmap(features_used, "Features Used", 'C', 'Model Order', C, model_orders)
    return train_accs, test_accs, features_used

# Comparison of different windows for a given time duration classification
def periodogram_window_comparison(y_train, y_test, ica_train, ica_test, classification_duration, windows, freq_prec = 1):
    # Doesn't support overlapping windows yet
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    freqs = []
    
    var_array = []
    for idx, window in enumerate(windows):
        extra_args = {"window":window}
        X_train, y_train_tmp, freq = extract_psd_features(y_train, ica_train, 'Periodogram_PSD', extra_args , fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = classification_duration, frequency_precision = freq_prec, compute_multiple_segs_per_trial = True)
        X_test, y_test_tmp, freq = extract_psd_features(y_test, ica_test, 'Periodogram_PSD', extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = classification_duration, frequency_precision = freq_prec, compute_multiple_segs_per_trial = True)
        X_train_array.append(X_train)
        y_train_array.append(y_train_tmp)
        X_test_array.append(X_test)
        y_test_array.append(y_test_tmp)
        freqs.append(freq)
        var_array.append(window)
           
    C = np.linspace(0.001, 1, 500)
    num_ica_comps = ica_train.shape[0]
    train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(np.asarray(X_train_array), np.asarray(X_test_array),
                                                        np.asarray(y_train_array), np.asarray(y_test_array), 
                                                            np.asarray(freqs), 'PSD', C, num_ica_comps, 
                                                                loss_fxn = 'squared_hinge', pen = 'l1')
    title = "Periodogram-{}s Classification for Various Windows".format(classification_duration)
    analyze_feature_performance('C', C, test_accs, train_accs, features_used, var_array, title, X_train_array[0].shape[1], metrics_computed = ['test'])

# Compares different Welch window lengths for a given classification time length duration
def welch_window_duration_comparison(y_train, y_test, ica_train, ica_test, window_duration, welch_window_durations, window='boxcar', frequency_precision = 1 ):
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    freqs = []
    
    var_array = []
    extra_args = {"window":window}
    sampling_freq = 250
    for idx, welch_dur in enumerate(welch_window_durations):
        extra_args["n_per_window"] =  (int)(welch_dur * sampling_freq)
        X_train, y_train_tmp, freq = extract_psd_features(y_train, ica_train, 'Welch_PSD', extra_args , fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = window_duration, frequency_precision = frequency_precision, compute_multiple_segs_per_trial = True)
        X_test, y_test_tmp, freq = extract_psd_features(y_test, ica_test, 'Welch_PSD', extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = window_duration, frequency_precision = frequency_precision, compute_multiple_segs_per_trial = True)
        X_train_array.append(X_train)
        y_train_array.append(y_train_tmp)
        X_test_array.append(X_test)
        y_test_array.append(y_test_tmp)
        freqs.append(freq)
        var_array.append("{}s".format(welch_dur))

    C = np.linspace(0.001, 0.1, 100)
    num_ica_comps = ica_train.shape[0]
    train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(np.asarray(X_train_array), np.asarray(X_test_array),
                                                    np.asarray(y_train_array), np.asarray(y_test_array), 
                                                        np.asarray(freqs), 'PSD', C, num_ica_comps, loss_fxn = 'squared_hinge', pen = 'l1')
    
    title = "{}s, 50% Overlap Welch-{} Window".format(window_duration, window)
    analyze_feature_performance('C', C, test_accs, train_accs, features_used, var_array, title, X_train_array[0].shape[1])

def psd_classification_wrt_data_points_analysis(y_train, y_test, ica_train, ica_test, method = "Periodogram_PSD", window = 'boxcar', AR_model_order = 22, frequency_prec = 1):
    max_time = 6
    min_time = 4
    sampling_freq = 250
    
    # C's chosen around optimal values using other experiments
    #C = np.linspace(0.01, 0.03, 100)
    
    C = np.linspace(0.001, 0.1, 100)
    
    N = np.arange(1, (int)((max_time-min_time)*sampling_freq)+1)
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    freqs = []
    if method == "Periodogram_PSD" or method == "Welch_PSD":  
        extra_args = {"window":window}
    elif method == "AR_Yule-Walker_PSD":
        extra_args = {"AR_model_order": AR_model_order}
        N = np.arange(30, (int)((max_time-min_time)*sampling_freq)+1)
        
    for idx, num_dp in enumerate(N):
        classification_duration = num_dp/sampling_freq
        X_train, y_train_tmp, freq = extract_psd_features(y_train, ica_train, method, extra_args , fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = classification_duration, frequency_precision = frequency_prec, compute_multiple_segs_per_trial = False)
        X_test, y_test_tmp, freq = extract_psd_features(y_test, ica_test, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                          sampling_freq = 250, window_duration = classification_duration, frequency_precision = frequency_prec, compute_multiple_segs_per_trial = False)
        X_train_array.append(X_train)
        y_train_array.append(y_train_tmp)
        X_test_array.append(X_test)
        y_test_array.append(y_test_tmp)
        freqs.append(freq)
           
    num_ica_comps = ica_train.shape[0]
    train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(np.asarray(X_train_array), np.asarray(X_test_array),
                                                        np.asarray(y_train_array), np.asarray(y_test_array), 
                                                            np.asarray(freqs), 'PSD', C, num_ica_comps, 
                                                                loss_fxn = 'squared_hinge', pen = 'l1')
    
    mean_test_accs = test_accs.mean(axis=1)
    if method == "Periodogram_PSD":
        dp_ticks = np.asarray([1, 125, 250, 375, 500])
        freq_res_ticks = np.round(sampling_freq/dp_ticks, decimals=2)
        title = "Periodogram (Boxcar Window) Mean Test Classification Accuracy For Various Data Lengths"
        y_label = "Mean Test Classification Accuracy (%)"
        x_label1 = "Number of Data Points"
        x_label2 = "Frequency Resolution (Hz)"
        plotter.plot_scatter_2_independent_vars(mean_test_accs, N, dp_ticks, freq_res_ticks, title, y_label, x_label1, x_label2)
    return mean_test_accs, N

def compare_AR_model_order_vs_test_class_accuracy(y_train, y_test, ica_train, ica_test, method, classification_durations):
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    
    feature_labels = []
    var_array = []
    
    model_orders = np.arange(1, 100, 1)
    mean_accs_array = []
    C = np.linspace(0.001, 0.1, 100)
    
    for j, dur in enumerate(classification_durations):
        var_array.append("{}s".format(dur))
        for idx, model_order in enumerate(model_orders):
            extra_args = {"AR_model_order": model_order}            
            X_train, y_train_tmp, freqs = extract_psd_features(y_train, ica_train, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = dur, frequency_precision = 1, compute_multiple_segs_per_trial = False)
            X_test, y_test_tmp, freqs = extract_psd_features(y_test, ica_test, method, extra_args, fft_length = 1024, min_time = 4, max_time = 6, 
                                                sampling_freq = 250, window_duration = dur, frequency_precision = 1, compute_multiple_segs_per_trial = False)
            feature_labels.append(freqs)
            X_train_array.append(X_train)
            y_train_array.append(y_train_tmp)
            X_test_array.append(X_test)
            y_test_array.append(y_test_tmp)
        
        num_ica_comps = ica_train.shape[0]
        train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(X_train_array, X_test_array,
                                                        y_train_array, y_test_array, feature_labels, "PSD", C, num_ica_comps, loss_fxn = 'squared_hinge', pen = 'l1')
        mean_test_accs = test_accs.mean(axis=1)
        mean_accs_array.append(mean_test_accs)
    title = "Using {}: AR Model vs. Test Classification Accuracy for Various Time Windows"
    analyze_feature_performance("Model Order", model_orders, mean_accs_array, [], [], var_array, title, 100, 
                                metrics_computed = ['test'])

if __name__ == "__main__":

    path = r'C:\Users\reyhanib\Documents\MATLAB\BCICompMI\A'
    directory = path + '1'
    
    #rejected_trials, eeg_train, eeg_test, eeg_trainfil, eeg_testfil, y_train, y_test = dl.load_dataset(directory)
    
    # Exclude rejected trials from train set
    #y_train = np.delete(y_train, rejected_trials)
    #eeg_trainfil = np.delete(eeg_trainfil, rejected_trials, axis=2)
    
    eeg_train, y_train, eeg_test, y_test = dl.load_pertinent_dataset(directory)
    # Run it for 3 class problems (hands and feet)
    y_train, eeg_train = pre.extract_3_class(y_train, eeg_train)
    y_test, eeg_test = pre.extract_3_class(y_test, eeg_test)
   # y_test, eeg_testfil = pre.extract_3_class(y_test, eeg_testfil)
    
    ica_test = pre.ica(directory, eeg_test)
    ica_train = pre.ica(directory, eeg_train)
   
    TEST_TYPE = 'periodogram_window_comparison'
    if TEST_TYPE == 'periodogram_dur_comparison':
        #print("periodogram")
        periodogram_window_durations = [2, 1, 0.5, .25]
        periodogram_classification_duration_comparison(y_train, y_test, ica_train, ica_test, periodogram_window_durations, window = 'boxcar')
    elif TEST_TYPE == 'welch_dur_comparison':  
        #print("welch")
        welch_window_durations = [1, 0.5, 0.25]
        classification_window_duration = 2
        welch_window_duration_comparison(y_train, y_test, ica_train, ica_test, classification_window_duration, welch_window_durations, window = 'kaiser (9)')
    elif TEST_TYPE == 'periodogram_window_comparison':
        #windows = ['boxcar', 'hamming' ,]
        windows = ['boxcar', 'hamming', 'kaiser (14)']
        #periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 2, windows, freq_prec = 1)
        periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 2, windows, freq_prec = 1)
        #periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 0.5, windows, freq_prec = 1)
        #periodogram_window_comparison(y_train, y_test, ica_train, ica_test, 0.25, windows, freq_prec = 1)
    elif TEST_TYPE == 'periodogram_dp_comparison':
        test_accs = psd_classification_wrt_data_points_analysis(y_train, y_test, ica_train, ica_test, window = 'boxcar', frequency_prec = 1)
    elif TEST_TYPE == 'AR_yule_walker_coeff_model_order_comparison':
        #ica_test = pre.ica(directory, eeg_testfil, algorithm='extended-infomax')
        train_accs, test_accs, features_used = AR_model_order_comparison(y_train, y_test, ica_train, ica_test, 1, 'AR_Yule-Walker_Coeffs', use_psd_features = False)
    elif TEST_TYPE == 'AR_yule_walker_psd_model_order_comparison':
        #ica_test = pre.ica(directory, eeg_testfil, algorithm='extended-infomax')
        train_accs, test_accs2, features_used = AR_model_order_comparison(y_train, y_test, ica_train, ica_test, 0.5, 'AR_Yule-Walker_PSD', use_psd_features = True)
    elif TEST_TYPE == 'AR_burg_psd_model_order_comparison':
        train_accs, test_accs2, features_used = AR_model_order_comparison(y_train, y_test, ica_train, ica_test, 2, 'AR_Burg_PSD', use_psd_features = True)
    elif TEST_TYPE == 'AR_covar_psd_model_order_comparison':
        train_accs, test_accs2, features_used = AR_model_order_comparison(y_train, y_test, ica_train, ica_test, 2, 'AR_Covar_PSD', use_psd_features = True)
    elif TEST_TYPE == 'AR_dp_comparison':
        model_orders = [11, 16, 21, 26]
        accs = []
        for idx, model_order in enumerate(model_orders):
            tmp_mean_accs, N = psd_classification_wrt_data_points_analysis(y_train, y_test, ica_train, ica_test, method = "AR_Yule-Walker_PSD", AR_model_order = model_order)
            accs.append(tmp_mean_accs)
        title = "Test Classification Accuracy Using AR-Yule Walker PSE for Various Data Points"
        analyze_feature_performance('Data Points (N)', N, accs, [], [], np.asarray(model_orders, str), title, 100, metrics_computed = ['test'])
    elif TEST_TYPE == 'AR_model_order_vs_test_acc':    
        test_accs, model_orders = compare_AR_model_order_vs_test_class_accuracy(y_train, y_test, ica_train, ica_test, "AR_Yule-Walker_PSD", [2, 1, 0.5])