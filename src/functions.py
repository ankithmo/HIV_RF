import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd
import pydotplus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.externals.six import StringIO
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import sys
sys.path.append("../.")

from variables import Variables
from setup_logger import logger

class Helpers:
  """
    Private functions
  """
  v = Variables()

  def error(self):
    """
      Print onto the console a message if an error is encountered during execution

      Arguments:
        - No arguments

      Returns:
        - No return values
    """
    print("Execution terminated. Check {0} for details".format(self.v.log_f))
    sys.exit(1)

  def create_file(self, filename):
    """
      Create a file if it does not exist

      Arguments:
        - filename: str
            Absolute path of the file that has to be created

      Returns:
        - No return values
    """
    try:
      os.path.exists(filename)
    except IOError:
      logger.info("Attempting to create {0}".format(filename))
      try:
        f = open(filename,'w')
        f.close()
      except IOError:
        logger.error("Unable to create {0}".format(filename))
        self.error()
      logger.info("{0} successfully created".format(filename))
    return None

  def done(self):
    """
      Print onto the console a message that execution has completed

      Arguments:
        - No arguments

      Returns:
        - No return values
    """
    print("Execution completed")
    return None

  def check_file_existence(self, filename):
    """
      Check if a file exists and throw error if not

      Arguments:
        - filename: str
            Absolute path of the file whose existence is to be determined

      Returns:
        - No return values
    """
    try:
      assert(os.path.exists(filename))
    except AssertionError:
      logger.error("{0} does not exist".format(filename))
      self.error()
    return None

  def check_year(self, start, end):
    """
      Check if end year is greater than the start year

      Arguments:
        - start: int
            Starting year
        - end: int
            Ending year
    
      Returns:
        - No return values
    """
    self.check_integer(start)
    self.check_integer(end)
    try:
      assert(end > start)
    except AssertionError:
      logger.error("End year must be greater than start year")
      self.error()
    return None

  def check_extension(self, filename, ext):
    """
      Validates the extension of the file

      Arguments:
        - filename: str
            Name of the file
        - ext: str
            Extension of the file

      Returns:
        - No return values
    """
    try:
      current = filename.split('.')[-1]
      assert(current == ext)
    except AssertionError:
      logger.error("Expected extension {0}, got {1}".format(ext, current))
      self.error()

  def import_dataset(self, abs_path):
    """
      Imports the dataset

      Arguments:
        - abs_path: str
            Absolute path to the dataset file

      Returns:
        - data: pandas dataframe
            Dataset as a pandas dataframe
    """
    self.check_file_existence(abs_path)
    logger.info("Reading {0}".format(abs_path))
    ext = abs_path.split('.')[-1]
    if ext == 'csv':
      data = pd.read_csv(abs_path)
    elif ext == 'xls':
      data = pd.read_excel(abs_path)
    else:
      logger.error("Expected extensions csv or xls, got {0}".format(ext))
      self.error()
    logger.info("{0} has {1} rows and {2} columns".format(abs_path, data.shape[0], data.shape[1]))
    return data

  def write_dataset(self, dataset, abs_path):
    """
      Writes dataset to a file

      Arguments:
        - dataset: pandas dataframe
            Dataset to be written

        - abs_path: str
            Absolute path to the file where the dataset must be written

      Returns:
        - No return value
    """
    logger.info("Creating {0}".format(abs_path))
    ext = abs_path.split('.')[-1]
    if ext == 'csv':
      dataset.to_csv(abs_path, index = False)
    elif ext == 'xls':
      dataset.to_excel(abs_path, index = False)
    else:
      logger.error("Expected extensions csv or xls, got {0}".format(ext))
      self.error()
    logger.info("{0} successfully generated".format(abs_path))
    return None

  def read_feature_file(self, filename):
    """
      read the features in filename

      Arguments:
        - filename: str
            Absolute path to the file where the features are newline separated strings

      Returns:
        - features: list
            List of features
    """
    with open(filename,'r') as f:
      features = f.readlines()
      features = [d.split('\n')[0] for d in features]
    return features

  def isfloat(self, inp):
    """
      Check if input is a floating-point number or not

      Arguments:
        - inp: Input 

      Returns:
        True if floating-point number, False otherwise
    """
    try:
      float(inp)
      return True
    except ValueError:
      return False

  def check_integer(self, number):
    """
      Ensure that input is a number

      Arguments:
        - number: Input

      Returns:
        - No return values
    """
    try:
      assert(str(number).isnumeric())
    except AssertionError:
      logger.error("Expected type {0}, got type {1} for {2}".format(type(number), 'int', number))
      self.error()

  def check_float(self, number):
    """
      Ensure that input is a floating-point number

      Arguments:
        - number: Input

      Returns:
        - No return values
    """
    try:
      assert(self.isfloat(number))
    except AssertionError:
      logger.info("Expected type {0}, got type {1} for {2}".format(type(number), 'float', number))
      self.error()

  def confmat_heatmap(self, cm, score, path):
    """
      Generate the heatmap for the confusion matrix

      Arguments:
        - cm: array
            confusion matrix
        - score: float
            Accuracy
        - path: str
            Absolute path to the directory where the heatmap will be generated

      Returns:
        - No return values
    """
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('True label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig(path)
    logger.info("{0} successfully generated".format(path))
    return None

  def roc_auc(self, fpr, tpr, auc, path):
    """
      Generate the ROC AUC curve 

      Arguments:
        - fpr: float
            False positive rates
        - tpr: float
            True positive rates
        - auc: float
            ROC AUC score
        - path: str
            Absolute path to the directory where the curve must be saved

      Returns:
        - No return values
    """
    plt.figure(figsize=(9,9))
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve with AUC: {0}'.format(auc), size=15)
    plt.savefig(path)
    logger.info("{0} successfully generated".format(path))
    return None

  def check_dir(self, path):
    """
      Check if directory is present and create it if not present

      Arguments:
        - path: str
            Absolute path to the directory 

      Returns:
        - No return values
    """
    if not os.path.exists(path):
      try:
        logger.info("Attempting to create {0} directory".format(path))
        os.makedirs(path)
        logger.info("{0} directory successfully created".format(path))
      except OSError:
        logger.error("Unable to create {0} directory".format(path))
        self.error()

  def get_metrics(self, model, data_list, results_path, accuracy=True, class_report=True, confmat=True, roc_auc=True):
    """
      Compute the accuracy, classification report, confusion matrix and ROC AUC curve

      Arguments:
        - model: Model whose metrics must be computed
        - data_list: list, length = 2 or 4 
            Dataset [X, Y]
            or
            Training and testing datasets [X_train, Y_train, X_test, Y_test]
        - results_path: str
            Absolute path to the directory where the metric files must be saved
        - accuracy: bool, optional, default = True
            Whether or not to log accuracy
        - class_report: bool, optional, default = True
            Whether or not to generate class_report_train.txt and class_report_test.txt
        - confmat: bool, optional, default = True
            Whether or not to generate confmat_train.png and confmat_test.png
        - roc_auc: bool, optional, default = True:
            Whether or not to generate roc_auc_train.png and roc_auc_test.png

      Returns:
        - No return values
    """
    if len(data_list) == 2:
      X, Y = data_list
      score = model.score(X, Y)
      pred = model.predict(X)
      if accuracy:
        logger.info("accuracy = {0}".format(score))
      if class_report:
        cr = classification_report(Y, pred)
        cr_file = os.path.join(results_path,'class_report.txt')
        self.create_file(cr_file)
        with open(cr_file, 'w') as f:
          f.write(cr)
        logger.info("Classification report generated: {0}".format(cr_file))
      if confmat:
        cm = confusion_matrix(Y, pred)
        self.confmat_heatmap(cm, score, os.path.join(results_path,'confmat.png'))
      if roc_auc:
        fpr, tpr, _ = roc_curve(Y, model.predict_proba(X)[:,1])
        auc_score = auc(fpr, tpr)
        logger.info("ROC AUC = {0}".format(auc_score))
        self.roc_auc(fpr, tpr, auc_score, os.path.join(results_path,'roc_auc.png'))
    elif len(data_list) == 4:
      X_train, Y_train, X_test, Y_test = data_list
    
      train_score = model.score(X_train, Y_train)
      test_score = model.score(X_test, Y_test)
    
      train_pred = model.predict(X_train)
      test_pred = model.predict(X_test)
    
      if accuracy:
        logger.info("Training accuracy = {0}".format(train_score))
        logger.info("Testing accuracy = {0}".format(test_score))

      if class_report:
        train_cr = classification_report(Y_train, train_pred)
        train_cr_file = os.path.join(results_path,'class_report_train.txt')
        self.create_file(train_cr_file)
        with open(train_cr_file,'w') as f:
          f.write(train_cr)
        logger.info("Training classification report generated: {0}".format(train_cr_file))

        test_cr = classification_report(Y_test, test_pred)
        test_cr_file = os.path.join(results_path,'class_report_test.txt')
        self.create_file(test_cr_file)
        with open(test_cr_file,'w') as f:
          f.write(test_cr)
        logger.info("Testing classification report generated: {0}".format(test_cr_file))

      if confmat:
        train_cm = confusion_matrix(Y_train, train_pred)
        self.confmat_heatmap(train_cm, train_score, os.path.join(results_path,'confmat_train.png'))
      
        test_cm = confusion_matrix(Y_test, test_pred)
        self.confmat_heatmap(test_cm, test_score, os.path.join(results_path,'confmat_test.png'))

      if roc_auc:
        train_fpr, train_tpr, _ = roc_curve(Y_train, model.predict_proba(X_train)[:,1])
        train_auc = auc(train_fpr, train_tpr)
        logger.info("Training ROC AUC = {0}".format(train_auc))
        self.roc_auc(train_fpr, train_tpr, train_auc, os.path.join(results_path,'roc_auc_train.png'))
      
        test_fpr, test_tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
        test_auc = auc(test_fpr, test_tpr)
        logger.info("Testing ROC AUC = {0}".format(test_auc))
        self.roc_auc(test_fpr, test_tpr, test_auc, os.path.join(results_path,'roc_auc_test.png'))
    else:
      logger.error("expected list of length 2 or 4, got {0}".format(len(data_list)))
      self.error()
    return None

  def decision_tree_viz(self, model, path):
    """
      Generate visualization of decision tree

      Arguments:
        - model: Decision tree model
        - path: str
            Absolute path to the file where the decision tree visualization is to be generated

      Returns:
        - No return values
    """
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = self.read_feature_file(self.v.features_f), class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    self.check_extension(path, 'png')
    graph.write_png(path)
    logger.info("{0} successfully generated".format(path))
    return None

  def get_one_roc_curve(self, data_list, results_path, logit, dec_tree, neural_net):
    """
      Generate all ROC curves in one plot

      Arguments:
        - data_list: list, length = 2
            Test set [X_test,Y_test]
        - results_path: str
            Absolute path to the directory where the ROC curve must be generated
        - data_list: list, length = 4
            Training and testing datasets [X_train, Y_train, X_test, Y_test]
        - logit: Logistic Regression model
        - dec_tree: Decision tree model
        - neural_net: Neural network model

      Returns:
        - No return value
    """
    if len(data_list) != 2:
      logger.error("Expected array of length 2, got {0}".format(len(data_list)))
      self.error()
    
    plt.figure(figsize=(9,9))

    X, Y = data_list
    self.check_file_existence(X)
    self.check_file_existence(Y)

    X = self.import_dataset(X)
    Y = self.import_dataset(Y)

    if logit != None:
      logit_fpr, logit_tpr, _ = roc_curve(Y, logit.predict_proba(X)[:,1])
      logit_auc_score = auc(logit_fpr, logit_tpr)
      plt.plot(logit_fpr, logit_tpr, color='red', lw=2, label='Logistic regression, AUC = {0}'.format(logit_auc_score))
    
    if dec_tree != None:
      dec_tree_fpr, dec_tree_tpr, _ = roc_curve(Y, dec_tree.predict_proba(X)[:,1])
      dec_tree_auc_score = auc(dec_tree_fpr, dec_tree_tpr)
      plt.plot(dec_tree_fpr, dec_tree_tpr, color='blue', lw=2, label='Decision tree, AUC = {0}'.format(dec_tree_auc_score))
    
    if neural_net != None:
      neural_net_fpr, neural_net_tpr, _ = roc_curve(Y, neural_net.predict_proba(X)[:,1])
      neural_net_auc_score = auc(neural_net_fpr, neural_net_tpr)
      plt.plot(neural_net_fpr, neural_net_tpr, color='green', lw=2, label='Neural network, AUC = {0}'.format(neural_net_auc_score))
    plt.plot([0, 1], [0, 1], color='yellow', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curves'.format(auc), size=15)
    plt.legend(loc='upper left')
    reqd_file = os.path.join(results_path, "final_roc.png")
    plt.savefig(reqd_file)
    logger.info("{0} successfully generated".format(reqd_file))
    return None
