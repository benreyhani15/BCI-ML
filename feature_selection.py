from sklearn.feature_selection import SelectPercentile, SelectKBest, RFE, f_classif, mutual_info_classif,  chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
import data_loader as dl
import preprocessing as pre
import plotter, classifier
import numpy as np
import pandas as pd
from feature_extraction import extract_psd_features, extract_coeff_features, extract_cwt_morlet_features
import matplotlib.pyplot as plt
from analyzer import analyze_feature_performance

def select_percent_using_stats(X_train, y_train, percent_keep, metric = 'ANOVA'):
    # metric can be 'ANOVA', 'MI', 'CHI2'
    score = f_classif
    if metric == 'MI':
        score = mutual_info_classif
    elif metric == 'CHI2':
        score = chi2
        
    select = SelectPercentile(score_func = score, percentile=percent_keep) 
    select.fit(X_train, y_train)
    return select.transform(X_train), select

def select_k_using_stats(X_train, y_train, num_features, metric = 'ANOVA'):
    # metric can be 'ANOVA', 'MI', 'CHI2'
    score = f_classif
    if metric == 'MI':
        score = mutual_info_classif
    elif metric == 'CHI2':
        score = chi2
        
    select = SelectKBest(score_func = score, k=num_features) 
    select.fit(X_train, y_train)
    return select.transform(X_train), select

def select_k_using_recursive(X_train, y_train, num_features, classifier):
    rfe = RFE(classifier, num_features, step = 1)
    selector = rfe.fit(X_train, y_train)
    return selector.transform(X_train), selector
                             
def feature_selection_stats_test(y_train, y_test, X_train, X_test, freqs, ar_model, use_percents = True):
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    freqs = []
    var_array = []
    if use_percents:
        params = [75, 50, 25, 10, 5]
    else:
        params = [500, 250, 100, 50, 25]
    metrics = ['ANOVA']
    
    X_train_array.append(X_train)
    y_train_array.append(y_train)
    X_test_array.append(X_test)
    y_test_array.append(y_test)
    freqs.append(freq)
    var_array.append("All 858 Features")
    
    for metric in metrics:
        for value in params:
            if use_percents:
                X_train_tmp, selector = select_percent_using_stats(X_train, y_train, value, metric = metric)
                var_array.append("{} using {}% of features".format(metric, value))
            else:
                X_train_tmp, selector = select_k_using_stats(X_train, y_train, value, metric = metric)
                var_array.append("{} using {} features".format(metric, value))
            X_test_tmp = selector.transform(X_test)
            print("Features reduced to: train: {}, test: {}".format(X_train_tmp.shape, X_test_tmp.shape))
            X_train_array.append(X_train_tmp)
            y_train_array.append(y_train)
            X_test_array.append(X_test_tmp)
            y_test_array.append(y_test)
            freqs.append(freq)
        
    C = np.linspace(0.001, .1, 100)
    #C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    #C = [ 1, 2.5, 5, 10, 20, 30, 50, 100, 500, 1000]
    num_ica_comps = ica_train.shape[0]

    train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(X_train_array, X_test_array,
                                                        y_train_array, y_test_array, 
                                                            freqs, 'PSD', C, num_ica_comps, 
                                                                loss_fxn = 'squared_hinge', pen = 'l1')
    title = "Test Classification Accuracy Using ANOVA For Feature Selection of 2s AR Burg with order: {}".format(ar_model)
    analyze_feature_performance('C', C, test_accs, train_accs, features_used, var_array, title, X_train_array[0].shape[1])
    return test_accs

def feature_selection_RFE_test(y_train, y_test, X_train, X_test, freqs):
    X_train_array = []
    y_train_array = []
    X_test_array = []
    y_test_array = []
    freqs = []
    var_array = []
    params = [500, 300, 200, 100, 50, 30]
    classifiers = [LinearDiscriminantAnalysis(), LinearSVC(penalty = 'l1'), LinearSVC(penalty = 'l2')]
    classifier_label = ["LDA", "Lin-SVM-l1", "Lin-SVM-l2"]
    X_train_array.append(X_train)
    y_train_array.append(y_train)
    X_test_array.append(X_test)
    y_test_array.append(y_test)
    freqs.append(freq)
    var_array.append("all features")
    
    for idx, classifier in enumerate(classifiers):
        for value in params:
            X_train_tmp, selector = select_k_using_recursive(X_train, y_train, value, classifier)
            var_array.append("{} using {} features".format(classifier_label[idx], value))
            X_test_tmp = selector.transform(X_test)
            print("Features reduced to: train: {}, test: {}".format(X_train_tmp.shape, X_test_tmp.shape))
            X_train_array.append(X_train_tmp)
            y_train_array.append(y_train)
            X_test_array.append(X_test_tmp)
            y_test_array.append(y_test)
            freqs.append(freq)
        
    C = np.linspace(0.001, .1, 100)
    #C = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    num_ica_comps = ica_train.shape[0]

    train_accs, test_accs, features_used = classifier.evaluate_multiple_linsvms_for_comparison(X_train_array, X_test_array,
                                                        y_train_array, y_test_array, 
                                                            freqs, 'PSD', C, num_ica_comps, 
                                                                loss_fxn = 'squared_hinge', pen = 'l1')
    title = "Periodogram with feature selection"
    analyze_feature_performance('C', C, test_accs, train_accs, features_used, var_array, title, X_train_array[0].shape[1])
    return test_accs

if __name__ == '__main__':
    path = '/Users/benreyhani/Files/GradSchool/BCISoftware/main/BCI/Dataset/A'
    directory = path + '1'
    eeg_train, y_train, eeg_test, y_test = dl.load_pertinent_dataset(directory)
    
    # Run it for 3 class problems (hands and feet)
    y_train, eeg_train = pre.extract_3_class(y_train, eeg_train)
    y_test, eeg_test = pre.extract_3_class(y_test, eeg_test)
    
    ica_test = pre.ica(directory, eeg_test)
    ica_train = pre.ica(directory, eeg_train)
    
    extra_args = {"window":"boxcar"}
    #X_train, y_train, freq = extract_cwt_morlet_features(y_train, ica_train, freqs = np.arange(3, 41, 2), frequency_res = 6)
    #X_test, y_test, freq = extract_cwt_morlet_features(y_test, ica_test, freqs = np.arange(3, 41, 2), frequency_res = 6)
    
    #X_train, y_train, freq = extract_psd_features(y_train, ica_train, 'Periodogram_PSD', extra_args, window_duration = 1)
    #X_test, y_test, freq = extract_psd_features(y_test, ica_test, 'Periodogram_PSD', extra_args, window_duration = 1)
    

    extra_args["AR_model_order"] = 28 
    X_train, y_train, freq = extract_psd_features(y_train, ica_train, 'AR_Burg_PSD', extra_args, window_duration = 2)
    X_test, y_test, freq = extract_psd_features(y_test, ica_test, 'AR_Burg_PSD', extra_args, window_duration = 2)
    test_accs = feature_selection_stats_test(y_train, y_test, X_train, X_test, freq, order, use_percents = False)

'''
    METHOD = "Stats"
    
    if METHOD == "Stats":
        test_accs = feature_selection_stats_test(y_train, y_test, X_train, X_test, freq, use_percents = False)
    elif METHOD == "RFE":
        test_accs = feature_selection_RFE_test(y_train, y_test, X_train, X_test, freq)
    '''