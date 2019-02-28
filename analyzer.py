import numpy as np
from sklearn.metrics import cohen_kappa_score
import plotter
import utils
import matplotlib.pyplot as plt
import pandas as pd

SUPPRESS_OUTPUT = False

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

def evalulate_classifier(title, classifier, X_test, y_test, X_train, y_train, feature_labels, feature_type, num_ica_comps, just_accuracy= True):
    predictions = classifier.predict(X_test)
    accuracy_test = classifier.score(X_test, y_test)
    accuracy_train = classifier.score(X_train, y_train)
    if SUPPRESS_OUTPUT == False:
        print("{}\nTest accuracy: {}\nTrain accuracy: {}".format(title,accuracy_test, accuracy_train))  
        if just_accuracy == False:
            plotter.plot_confusion_matrix(["Left", "Right", "Foot"], y_test, predictions)
            print("Cohen Kappa Score: {}".format(cohen_kappa_score(predictions, y_test)))
            find_useful_features(classifier.coef_)
    return accuracy_test, accuracy_train
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
    useless_components = ""
    for i in np.arange(num_ica_comps):
        features_per_comp = freqs.shape[0]
        start_idx = i*features_per_comp
        
        components_binary = binary_importance[start_idx:(start_idx+features_per_comp)]
        important_freqs = freqs[np.argwhere(components_binary == 1)]
        if len(important_freqs > 0):         
            print("For component #{}, important features are at frequencies: {}".format(i, important_freqs))
            plotter.plot_binary_scatter(freqs, components_binary, "Component #{}".format(i), important_freqs)
           
            min_freq = important_freqs.min()
            min_relevant_freq = min_freq if min_freq < min_relevant_freq else min_relevant_freq
            max_freq = important_freqs.max()
            max_relevant_freq = max_freq if max_freq > max_relevant_freq else max_relevant_freq
        else:
            useless_components+=", {}".format(i)
    summary = "Minimum Important freq: {}, Max important freq: {}, Useless components: {}".format(min_relevant_freq, max_relevant_freq, useless_components)
    print(summary)
    
def get_feature_importances(svm, classes, num_ica_comps, feature_type, feature_labels):
    weights = svm.coef_
    feature_importance = np.abs(weights)
    ica_comps = np.tile(np.repeat(np.arange(num_ica_comps), len(feature_labels)), len(classes))
    features = np.tile(np.tile(feature_labels, num_ica_comps), len(classes))
    labels = []
    targets = []
    for index, class_label in enumerate(classes):
        labels.append(np.repeat(class_label, feature_importance.shape[1]))
        targets.append(np.repeat(index+1, feature_importance.shape[1]))
    labels = np.asarray(labels).flatten()
    targets = np.asarray(targets).flatten()
    sums = np.reshape(feature_importance.sum(axis=1), (feature_importance.shape[0], 1))
    percent_importance = (np.divide(feature_importance, sums) * 100).flatten()
    pandas_dict = {"Class label": labels, "Targets" : targets, "ICA Comp" : ica_comps, feature_type: features, "Feature Importance":feature_importance.flatten(), "Percent Importance" : percent_importance}
    print("class labels: {}, targets: {}, ica_comps: {}, feature_type: {}, feature_importances: {}".format(labels.shape, targets.shape, ica_comps.shape, features.shape, feature_importance.flatten().shape))
    return pd.DataFrame(pandas_dict)

def extract_rows_from_pd_df_column(data_frame, column_key, column_value):
    return data_frame.loc[data_frame[column_key] == column_value]

def find_useful_features(weights):
     means = weights.mean(axis=0)
     idx_meanzeroes = np.argwhere(means == 0)
     var = weights.var(axis = 0)
     idx_varzeroes = np.argwhere(var == 0)
     # Feature indices that had no effect on any of the classifiers:
     useless_features = np.intersect1d(idx_meanzeroes, idx_varzeroes)
     useful_features = np.setdiff1d(np.arange(len(means)), useless_features)
     if SUPPRESS_OUTPUT == False:
         print("useless_features size: {}, useful_features: {} feature size: {}".format(len(useless_features), len(useful_features),len(means)))
     return useful_features, useless_features

def analyze_feature_performance(independent_var_label, independent_var, test_accs, train_accs, features_used, variable_array, title, orig_feature_count, 
                                metrics_computed = ['test', 'train', 'features']):
    if 'test' in metrics_computed: df_test_acc = pd.DataFrame({independent_var_label: independent_var})
    if 'train' in metrics_computed: df_train_acc = pd.DataFrame({independent_var_label: independent_var})
    if 'features' in metrics_computed: df_features = pd.DataFrame({independent_var_label: independent_var})
    
    for i, content in enumerate(variable_array):
        if 'test' in metrics_computed: df_test_acc[content] = test_accs[i]
        if 'train' in metrics_computed: df_train_acc[content] = train_accs[i]
        if 'features' in metrics_computed: df_features[content] = features_used[i]
    
    if 'test' in metrics_computed:
        print("Test Classification")
        plotter.plot_multivariable_scatter(df_test_acc, 'Classification Accuracy (%)', r"{}: Test Classification Accuracy".format(title))
        for i, content in enumerate(variable_array):
            print("{} Stats: mean={:.2f}%, max={:.2f}%".format(content, test_accs[i].mean(), test_accs[i].max()))
    
    if 'train' in metrics_computed:
        print("Train Classification")
        plotter.plot_multivariable_scatter(df_train_acc, 'Classification Accuracy (%)', r"{}: Train Classification Accuracy".format(title))
        for i, content in enumerate(variable_array):
            print("{} Stats: mean={:.2f}%, max={:.2f}%".format(content, train_accs[i].mean(), train_accs[i].max()))
    
    if 'features' in metrics_computed:
        print("Features Used")
        plotter.plot_multivariable_scatter(df_features, 'Number of Features Used', r"{}: Fraction of {} Original Features Used".format(title, orig_feature_count))
        for i, content in enumerate(variable_array):
            print("{} Stats: max={}, min={}".format(content, features_used[i].max(), features_used[i].min()))
            
def test_run_time(sphere, weights, datum, scaler, classifier):
    import datetime
    import time
    import feature_extraction as fe
    start = datetime.datetime.now()
    ica = np.dot(weights, np.dot(sphere, datum)).reshape((22, 1750, 1))
    X, f = fe.compute_psdp(ica)
    X = scaler.transform(X)
    classifier.predict(X)[0]
    end = datetime.datetime.now()
    c = end-start
    return c    