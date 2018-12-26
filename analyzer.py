import numpy as np
from sklearn.metrics import cohen_kappa_score
import plotter
import utils
import matplotlib.pyplot as plt

def display_score_matrix(grid_search):
    print("\n\nBest params for this grid: {}".format(grid_search.best_params_))
    print("\nBest CV mean accuracy: {}".format(grid_search.best_score_))
    print(get_string_results(grid_search))
    
def get_string_results(grid_search):
    string = ""
    for i, item in enumerate(grid_search.cv_results_['params']):
        string +="\n{} ; mean: {} ; std: {}".format(item, 
               grid_search.cv_results_['mean_test_score'][i],
                   grid_search.cv_results_['std_test_score'][i])
    return string

def evalulate_classifier(title, classifier, X_test, y_test, freqs, num_ica_comps):
    predictions = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)
    #plotter.plot_confusion_matrix(["Left", "Right", "Foot", "Tongue"], y_test, predictions)
    print("{}\nClassification accuracy on test set: {}".format(title,accuracy))
    #print("Cohen Kappa Score: {}".format(cohen_kappa_score(predictions, y_test)))
    return accuracy
    #analyze_features(classifier.coef_, freqs, num_ica_comps)
   # analyze_classifier_coeffs(classifier.coef_, freqs, num_ica_comps)
   # evaluate_individual_decisions(classifier.decision_function(X_test, y_test))
    
   #def evaluate_individual_decisions(decision_probs):
#TODO: Graphically show the decision probs for each target
   
def analyze_features(weights, freqs, num_ica_comps):
    useful_features, useless_features = find_useful_features(weights)
    binary_importance = np.ones(weights.shape[1])
    binary_importance[useless_features] = 0    
    max_relevant_freq = 0
    min_relevant_freq = 100
    for i in np.arange(num_ica_comps):
        features_per_comp = freqs.shape[0]
        start_idx = i*features_per_comp
        
        components_binary = binary_importance[start_idx:(start_idx+features_per_comp)]
        important_freqs = freqs[np.argwhere(components_binary == 1)]
        print("For component #{}, important features are at frequencies: {}".format(i, important_freqs))
        plotter.plot_binary_scatter(freqs, components_binary, "Component #{}".format(i), important_freqs)
       
        min_freq = important_freqs.min()
        min_relevant_freq = min_freq if min_freq < min_relevant_freq else min_relevant_freq
        max_freq = important_freqs.max()
        max_relevant_freq = max_freq if max_freq > max_relevant_freq else max_relevant_freq
    summary = "Minimum Important freq: {}, Max important freq: {}".format(min_relevant_freq, max_relevant_freq)
    print(summary)
    
              
def find_useful_features(weights):
     means = weights.mean(axis=0)
     idx_meanzeroes = np.argwhere(means == 0)
     var = weights.var(axis = 0)
     idx_varzeroes = np.argwhere(var == 0)
     # Feature indices that had no effect on any of the classifiers:
     useless_features = np.intersect1d(idx_meanzeroes, idx_varzeroes)
     useful_features = np.setdiff1d(np.arange(len(means)), useless_features)
     print("useless_features size: {}, useful_features: {} feature size: {}".format(len(useless_features), len(useful_features),len(means)))
     return useful_features, useless_features