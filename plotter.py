from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from numpy.fft import fft, fftshift
import pandas as pd

def plot_confusion_matrix(classes, actual, predictions):
    cm = confusion_matrix(actual, predictions)
    #print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
def plot_binary_scatter(x, y, title, freqs):
    plt.figure()
    plt.scatter(x, y, s=5)
    plt.title(title)
    freqs = np.around(freqs, 2)
    string_text = ""
    for i, freq in enumerate(freqs[:,0]):
        string_text+= " {}, ".format(freq) if i%8 != 0 else "\n{}, ".format(freq)
    plt.text(1, 0.1, string_text, ha='left', wrap=True)
    plt.show()
    #return plt

def plot_2d_annotated_heatmap(data, title, xlabel, ylabel, x_vals, y_vals):
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticklabels(y_vals)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            text = ax.text(j, i, data[i, j],
                       ha="center", va="center", color="black")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()
    
def plot_scatter_2_independent_vars(dependent_var, independent_var, ticks1, ticks2, title, y_label, x_label1, x_label2):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.plot(independent_var, dependent_var)
    ax1.set_xlabel(x_label1)
    ax2.set_xlim(ax1.get_xlim())
    ax1.set_xticks(ticks1)
    ax2.set_xticks(ticks1)
    ax2.set_xticklabels(ticks2)
    ax2.set_xlabel(x_label2)
    plt.title(title, y=1.15)
    plt.show()
    
def plot_multivariable_scatter(df, ylabel, title):
    # Assume first column is df is independent value and xlabel
    plt.figure()
    xlabel = df.columns[0]
    for i in np.arange(1, len(df.columns)):    
        plt.plot(xlabel, df.columns[i], data=df, marker='o', markerfacecolor='green', markersize=6, linewidth=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_window(window, title, nfft, sampling_freq):
    # Time Domain
    plt.figure()
    plt.plot(window)
    plt.title("Time Domain of the {} Window with {} Data Points".format(title, len(window)))
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()
    
    # Frequency Response
    plt.figure()
    A = fft(window, nfft)/(len(window)/2)
    nyquist_limit = sampling_freq/2
    freq = np.linspace(-nyquist_limit, nyquist_limit, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    plt.plot(freq, response)
    plt.axis([-nyquist_limit, nyquist_limit, -80, 0])
    plt.title(r"Frequency Response of the {} Window with {} Data Points".format(title, len(window)))
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Frequency [Hz]")
    plt.show()