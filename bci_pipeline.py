import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import feature_extraction as fe
from feature_selection import select_k_using_stats
import preprocessing as pre
import pandas as pd
import scipy as sp
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
from joblib import dump, load
from mne.io import read_raw_brainvision
from time import sleep, time
from scipy.signal import periodogram
from network_sender import send_data_to_vr_cpu
import os

def extract_NF_labels(NF_marker_array):
    idx = np.where(NF_marker_array[:, 0] < 4)
    labels = NF_marker_array[idx, 0]
    return labels[0]

def extract_FB_labels(FB_marker_array):
    labels = np.abs(FB_marker_array[:, 0])
    return labels

def grid_search_cv_linsvm(Y, ica_data, folds = 5, repeats = 5, is_l1 = True,
                          features = [30, 50, 100, 250, 500, 750, 850],
                              C = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500], dur = 2,
                                  min_time = 0, max_time = 2):
    X_train, y_train, freq = fe.extract_psd_features(Y, ica_data, "Periodogram_PSD", {"window":"boxcar"}, window_duration = dur, min_time = min_time
                                                     , max_time = max_time)
    pipe = Pipeline(steps = [('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func = f_classif)), 
             ('classifier', LinearSVC(penalty = 'l1', loss = 'squared_hinge', dual = False, class_weight = 'balanced', max_iter = 100000000))])
    param_grid = {'feature_selection__k':features, 'classifier__C': C}
    metrics = {'accuracy':make_scorer(accuracy_score), 'kappa': make_scorer(cohen_kappa_score)}
    
    if repeats>1:
        kf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    else:
        kf = StratifiedKFold(n_splits = folds, shuffle = True)
        
    grid = GridSearchCV(pipe, param_grid, scoring = metrics, refit = 'kappa', cv = kf)
    grid.fit(X_train, y_train)
    cv_results = grid.cv_results_
    results_df = get_results_df(cv_results)
    print("Best params: {}".format(grid.best_params_))
    return grid, results_df
    
def get_results_df(cv_results):
    df = DataFrame(cv_results)
    df_pruned = df[['mean_test_accuracy', 'mean_test_kappa', 'mean_train_accuracy', 
                    'mean_train_kappa', 'param_classifier__C', 'param_feature_selection__k', 
                    'rank_test_accuracy', 'rank_test_kappa', 'std_test_accuracy', 'std_test_kappa']]
    return df_pruned

eeg_comp = False

#  have to change these directories for paths in code for EEG computer
analysis_directory_path = 'add' if eeg_comp else 'C:\\Users\\reyhanib\\Documents\\Python Scripts\\Dataset\\'
FB_file_name = "FB.vhdr" if eeg_comp else "FBMarkerTestPerf.vhdr"

is_verbose = False
# Ran from EEG computer for online session
def run_bci(participant_name, analysis_period = 1, block_dur = 2, sampling_freq = 250):
    pipe = load_bci_pipe(participant_name)
    fb_file = os.path.join(analysis_directory_path+participant_name, FB_file_name)
    ica_weights, ica_sphere = load_ica(participant_name)
    while True:
        sleep(analysis_period)
        if is_verbose: print("analyzing new block")
        eeg = load_eeg_data(fb_file)
        if eeg.shape[1] >= (sampling_freq*block_dur):
            block = eeg[:, -(sampling_freq*block_dur):]
            if is_verbose: print("block shape: {}".format(block.shape))
            ica_data = np.dot(ica_weights, np.dot(ica_sphere, block))
            if is_verbose: print("ica shape: {}".format(ica_data.shape))
            X, y, freq = fe.extract_psd_features(np.asarray([0]), np.reshape(ica_data, (ica_data.shape[0], ica_data.shape[1], 1)), "Periodogram_PSD", {"window":"boxcar"}, 
                                                 fft_length = 1024, min_time = 0, max_time = 2, sampling_freq = 250, window_duration = 2, frequency_precision = 1)
            if is_verbose: print("X shape: {}".format(X.shape))
            y = pipe.predict(X)[0]
            print("prediction: {}".format(y))
            send_data_to_vr_cpu(str(y))
            
def load_eeg_data(file):
     raw = read_raw_brainvision(file, stim_channel = False, verbose = False) 
     if is_verbose: print("Data points in file: {}".format(raw.n_times))
     eeg_data = raw.get_data()
     return eeg_data
 
def load_ica(participant_name):
    file_path = analysis_directory_path + participant_name
    ica_weights = sp.io.loadmat(os.path.join(file_path,'ica_weights'))['ica_weights']
    ica_sphere = sp.io.loadmat(os.path.join(file_path,'ica_sphere'))['ica_sphere']
    return ica_weights, ica_sphere

def import_dfs(participant_name):
    file_path = analysis_directory_path + participant_name
    eeg = sp.io.loadmat(os.path.join(file_path,'eeg'))['eeg']
    ica_weights, ica_sphere = load_ica(participant_name)
    labels_path = 'C:\\Users\\reyhanib\\Documents\\VR\\ReyVR\\VR-master\\Logs\\' + participant_name
    labels = np.loadtxt(os.path.join(labels_path,'labels.txt'), dtype = 'int32')
    print("Size of labels: {}, Size of EEG data points: {}".format(labels.shape, eeg.shape[2]))
    print("Class distribution: right={}, fwd={}, left={}".format(
            np.sum(labels==1), np.sum(labels==2), np.sum(labels==3)))
    ica_data = pre.ica_dot(eeg, ica_weights, ica_sphere)
    return labels, ica_data

def save_bci_pipe(pipe, participant_name):
    file_path = analysis_directory_path + participant_name
    file = os.path.join(file_path, 'bci_pipe.joblib')
    dump(pipe, file)
    
def load_bci_pipe(participant_name):
    file_path = analysis_directory_path + participant_name
    file = os.path.join(file_path, 'bci_pipe.joblib')
    pipe = load(file)
    return pipe

#Run after collecting AFB data from VR computer
def find_best_bci_pipe(name):
    Y, ica_data = import_dfs(name)
    grid, df_results = grid_search_cv_linsvm(Y, ica_data, folds = 5, repeats = 5, is_l1 = True,
                          features = [30, 50, 100, 250, 500, 750, 850],
                              C = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500], dur = 2,
                                  min_time = 0, max_time = 2)
    best_pipe = grid.best_estimator_ # based on kappa score
    save_bci_pipe(best_pipe, name)
    return best_pipe, grid, df_results

if __name__ == "__main__":
    name = "Ben"
    Y, ica_data = import_dfs(name)
    grid, df_results = grid_search_cv_linsvm(Y, ica_data, folds = 5, repeats = 5, is_l1 = True,
                          features = [30, 50, 100, 250, 500, 750, 850],
                              C = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500], dur = 2,
                                  min_time = 0, max_time = 2)
    best_pipe = grid.best_estimator_ # based on kappa score
    save_bci_pipe(best_pipe, name)