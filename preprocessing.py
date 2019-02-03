import numpy as np
import scipy as sp
from scipy import io
import os
from sklearn import preprocessing
import data_loader
from sklearn.feature_selection import VarianceThreshold

def compute_ica(path, eeg):
    sphere, ica_weights = data_loader.load_ica_mats(path)    
    ica_data = (np.dot(ica_weights, np.dot(sphere, eeg.reshape(eeg.shape[0], eeg.shape[1]*eeg.shape[2])))).reshape(ica_weights.shape[0], eeg.shape[1], eeg.shape[2])
    return ica_data

def check_trainica_comp(path, ica_computed, is_filtered):
    file = 'ica' if is_filtered else 'ica2'
    ica_actual = sp.io.loadmat(os.path.join(path,file))[file]
    print("size of ica_actual: {}".format(ica_actual.shape))
    print("size of ica_predict: {}".format(ica_computed.shape))
    residual = np.abs(ica_computed-ica_actual)
    print("ICA diff: mean: {}, max: {}, var: {}\n".format(residual.mean(), residual.max(), residual.var()))
    
def ica(path, eeg):
    ica_data = compute_ica(path, eeg)
    return ica_data

def standardise_data(data):
    scaler = preprocessing.StandardScaler().fit(data)
    standardized_data = scaler.transform(data)
    return standardized_data, scaler

def extract_3_class(labels, data):
    idx =  np.where(labels != 4)[0]
    labels = labels[idx]
    data = data[:, :, idx]
    return labels, data

def select_features_using_var(X, y, threshold = 0.0):
    selector = VarianceThreshold(threshold)
    X_reduced = selector.fit_transform(X, y)
    print("Feature's reduced using var from: {}, to: {}".format(X.shape[1], X_reduced.shape[1]))
    return selector