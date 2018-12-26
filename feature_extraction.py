import numpy as np
from scipy.signal import periodogram

def compute_psdp(data, fft_length = 1024, fft_window = "boxcar", min_time = 4, 
                 max_time = 6, sampling_freq = 250):
    features_array = []
    for i in np.arange(data.shape[2]):
        #print("\n\ncomputing PSDs for data sample: {}".format(i))
        tmp_array = []
        for j in np.arange(data.shape[0]):
       #     print("Working on component: {}".format(j))
            datum = data[j, min_time*sampling_freq : max_time*sampling_freq, i]
      #      print("Shape of datum: {}".format(datum.shape))
            f, p = periodogram(datum, fs=sampling_freq, window=fft_window, 
                           nfft=fft_length, detrend=False)
            idx = np.argwhere((f>=2) & (f<=40.25))[:,0]
            psd_features = p[idx]
     #       print("# of extracted features: {}".format(len(idx)))
            tmp_array.append(psd_features)
        features_array.append(tmp_array)
    psd_features = np.asarray(features_array)
    print("Shape of PSD features: {}\n".format(psd_features.shape))
    psd_features = psd_features.reshape(psd_features.shape[0], 
           psd_features.shape[1]*psd_features.shape[2])
    print("Final features shape: {}\n".format(psd_features.shape))
    return psd_features, f[idx]