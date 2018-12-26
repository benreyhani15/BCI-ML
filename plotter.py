from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

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