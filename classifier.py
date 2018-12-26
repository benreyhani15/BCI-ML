from sklearn.model_selection import GridSearchCV
from sklearn import svm
from utils import send_email_notification
import numpy as np
import analyzer
import preprocessing as pre

def kfold_cv(classifier, X_train, y_train, param_grid, send_notif, title, folds=10):
    grid_search = GridSearchCV(classifier, param_grid, cv=folds)
    grid_search.fit(X_train, y_train)
    analyzer.display_score_matrix(grid_search)
    if send_notif: 
        send_email_notification("{}\n\nResults for search: {}".format(title, analyzer.get_string_results(grid_search)))
    return grid_search
    
def find_linear_SVM(X_train, y_train, param_grid, pen, loss_fxn, send_notif, title, folds=10):
    X_train, scaler = pre.standardise_data(X_train)
    duals = True
    if pen == 'l1':
        duals = False
    lin_svm = svm.LinearSVC(penalty=pen, loss=loss_fxn, dual=duals, max_iter = 1000000)
    print(lin_svm)
    grid_search = kfold_cv(lin_svm, X_train, y_train, param_grid, send_notif, title, folds)
    return grid_search
    
def train_linear_SVM(X_train, y_train, loss_fxn, pen, c):
    duals = True
    if pen == 'l1':
        duals = False
    lin_svm = svm.LinearSVC(penalty = pen, loss = loss_fxn, dual = duals, C = c)
    lin_svm.fit(X_train, y_train)
    return lin_svm
    
def evaluate_linear_SVM(X_train, y_train, X_test, y_test, loss_fxn, penalty, c, freqs, num_ica_comps):
    X_train_standard, scaler = pre.standardise_data(X_train)
    lin_svm = train_linear_SVM(X_train_standard, y_train, loss_fxn, penalty, c)
    X_test_standard = scaler.transform(X_test)
    accuracy = analyzer.evalulate_classifier("Linear SVM with L1 and c: {}".format(c), lin_svm, X_test_standard, y_test, freqs, num_ica_comps)
    return accuracy