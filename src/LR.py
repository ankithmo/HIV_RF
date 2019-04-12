from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../.")

from functions import Helpers
from variables import Variables

def LR(input_list, results_path, seed=123, k_folds=10):
  """
    1. Perform k-fold logistic regression on X and Y
    2. Get heatmap of confusion matrix
    3. Get ROC curve

    Arguments:
      - input_list: Absolute path to [X,Y] or [X_train, Y_train, X_test, Y_test]
      - results_path: Absolute path to the directory where the figures must be saved
      - seed (Optional): Random seed
          Default: 123
      - k_folds (Optional): Number of folds for cross-validation
          Default: 10

    Returns:
      - Trained logistic regression model
  """

  h = Helpers()
  v = Variables()
  
  # Diagnostics
  h.check_dir(results_path)

  num_files = len(input_list)
  if num_files == 2:
    X,Y = input_list
    h.check_file_existence(X)
    h.check_file_existence(Y)
  elif num_files == 4:
    X_train,Y_train,X_test,Y_test = input_list
    h.check_file_existence(X_train)
    h.check_file_existence(Y_train)
    h.check_file_existence(X_test)
    h.check_file_existence(Y_test)
  else:
    print "{0} files found in input_list, expected 2 or 4".format(num_files)

  # Import datasets
  if num_files == 2:
    X = h.import_dataset(X)
    Y = h.import_dataset(Y)
  else:
    X_train = h.import_dataset(X_train)
    Y_train = h.import_dataset(Y_train)
    X_test = h.import_dataset(X_test)
    Y_test = h.import_dataset(Y_test)
  
  # Train LR model
  if num_files == 2:
    lr = LogisticRegressionCV(solver='liblinear', cv=k_folds, random_state=seed)
    lr.fit(X, Y)

    # accuracy
    score = lr.score(X, Y)
    print "accuracy =", score

    # build confusion matrix
    cm = confusion_matrix(Y, lr.predict(X))
    h.confmat_heatmap(cm, score, os.path.join(results_path,'lr_confmat.png'))

    # build roc auc curve
    fpr, tpr, _ = roc_curve(lr.predict(X), Y, drop_intermediate=False)
    auc = roc_auc_score(lr.predict(X), Y)
    print "roc auc =", auc
    h.roc_auc(fpr, tpr, auc, os.path.join(results_path,'lr_roc_auc.png'))
  else:
    lr = logisticregression(solver='liblinear', random_state=seed)
    lr.fit(X_train, Y_train)

    # accuracy
    train_score = lr.score(X_train, Y_train)
    print "Training accuracy =", train_score
    test_score = lr.score(X_test, Y_test)
    print "Testing accuracy =", test_score

    # build confusion matrix
    train_cm = confusion_matrix(Y_train, lr.predict(X_train))
    h.confmat_heatmap(train_cm, train_score, os.path.join(results_path,'lr_confmat_train.png'))
    test_cm = confusion_matrix(Y_test, lr.predict(X_test))
    h.confmat_heatmap(test_cm, test_score, os.path.join(results_path,'lr_confmat_test.png'))

    # build roc auc curve
    train_fpr, train_tpr, _ = roc_curve(lr.predict(X_train), Y_train, drop_intermediate=False)
    train_auc = roc_auc_score(lr.predict(X_train), Y_train)
    print "roc auc =", train_auc
    h.roc_auc(train_fpr, train_tpr, train_auc, os.path.join(results_path,'lr_roc_auc_train.png'))
    test_fpr, test_tpr, _ = roc_curve(lr.predict(X_test), Y_test, drop_intermediate=False)
    test_auc = roc_auc_score(lr.predict(X_test), Y_test)
    print "roc auc =", test_auc
    h.roc_auc(test_fpr, test_tpr, test_auc, os.path.join(results_path,'lr_roc_auc_test.png'))

  return lr
