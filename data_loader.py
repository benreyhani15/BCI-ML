from scipy import io
import os
import scipy as sp
import numpy as np

def load_EEGdataset_using_MAT(path):
    # Already epoched
    eeg_train = sp.io.loadmat(os.path.join(path,'traineeg'))['eeg']
    eeg_train = eeg_train[0:22]
    #Filtered from epochs 
    eeg_test_filtered = sp.io.loadmat(os.path.join(path,'testeeg_filt'))['eeg']
    eeg_test_filtered = eeg_test_filtered[0:22]  
    
    eeg_train_filtered, eeg_test, reject_idx = load_pertinentEEGdataset(path)
   
    return reject_idx, eeg_train, eeg_test, eeg_train_filtered, eeg_test_filtered

def load_pertinentEEGdataset(path):
     # Filtered from cont
    eeg_train_filtered = sp.io.loadmat(os.path.join(path,'traineeg_filt'))['eeg']
    eeg_train_filtered = eeg_train_filtered[0:22]
    
    eeg_test = sp.io.loadmat(os.path.join(path,'testeeg'))['eeg']
    eeg_test = eeg_test[0:22]  
    
    #reject_idx = sp.io.loadmat(os.path.join(path, 'reject_idx'))['reject_idx']
    return eeg_train_filtered, eeg_test

def load_EEGdataset_using_MNE(path):
    # Are not epoched or filtered : BROKEN FOR BCI COMP DATA SET (CANT EXTRACT EPOCHS PROPERLY)
    """y_train, y_test = load_labels(path)
    eeg_train = mne.io.read_raw_edf(os.path.join(path, 'EEG_train.gdf'), preload=True)
    eeg_test = mne.io.read_raw_edf(os.path.join(path, 'EEG_test.gdf'), preload = True)
    train_events = mne.find_events(eeg_train, stim_channel='STI 014')
    test_events = mne.find_events(eeg_test, stim_channel='STI 014')
    
    train_filtered 
    """
    
# Algorithm can take on values: 'extended-infomax', 'sobi', 'jade'
def load_ica_mats(path):
    sphere = sp.io.loadmat(os.path.join(path,'ica_sphere'))['ica_sphere']
    ica_weights = sp.io.loadmat(os.path.join(path, 'ica_weights'))['ica_weights']
    return sphere, ica_weights

def load_labels(path):
    y_train = io.loadmat(os.path.join(path, 'y_train'))['classlabel'][:,0]
    y_test = io.loadmat(os.path.join(path, 'y_test'))['classlabel'][:,0]
    # Left:1, Right:2, Foot:3, Tongue:4
    unique, counts = np.unique(y_train, return_counts = True)
    print("class distribution for train set: {}\n".format(dict(zip(unique, counts))))
    
    unique, counts = np.unique(y_test, return_counts = True)
    print("class distribution for test set: {}\n".format(dict(zip(unique, counts))))
    return y_train, y_test

def load_dataset(path):
    rejected_idx, eeg_train, eeg_test, eeg_train_filt, eeg_test_filt = load_EEGdataset_using_MAT(path)
    y_train, y_test = load_labels(path)
    return rejected_idx, eeg_train, eeg_test, eeg_train_filt, eeg_test_filt, y_train, y_test

def load_pertinent_dataset(path):
    eeg_train, eeg_test = load_pertinentEEGdataset(path)
    y_train, y_test = load_labels(path)
    return eeg_train, y_train, eeg_test, y_test