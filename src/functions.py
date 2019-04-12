import pandas as pd
import numpy as np
import os
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

class Helpers:
  """
    Private functions
  """
  def check_file_existence(self, filename):
    try:
      assert(os.path.exists(filename))
    except AssertionError:
      print "{0} does not exist".format(filename)
      sys.exit(1)

  def check_year(self, start, end):
    self.check_integer(start)
    self.check_integer(end)
    try:
      assert(end > start)
    except AssertionError:
      print "End year must be greater than start year"
      sys.exit(1)

  def check_extension(self, filename, ext):
    try:
      current = filename.split('.')[-1]
      assert(current == ext)
    except AssertionError:
      print "Expected extension {0}, got {1}".format(ext, current)
      sys.exit(1)

  def import_dataset(self, abs_path):
    print "Reading {0}".format(abs_path)
    ext = abs_path.split('.')[-1]
    if ext == 'csv':
      data = pd.read_csv(abs_path)
    elif ext == 'xls':
      data = pd.read_excel(abs_path)
    else:
      print "Expected extensions csv or xls, got {0}".format(ext)
      sys.exit(1)
    print "{0} has {1} rows and {2} columns".format(abs_path, data.shape[0], data.shape[1])
    return data

  def write_dataset(self, dataset, abs_path):
    print "Creating {0}".format(abs_path)
    ext = abs_path.split('.')[-1]
    if ext == 'csv':
      dataset.to_csv(abs_path, index = False)
    elif ext == 'xls':
      dataset.to_excel(abs_path, index = False)
    else:
      print "Expected extensions csv or xls, got {0}".format(ext)
      sys.exit(1)
    print "{0} successfully generated".format(abs_path)

  def normalize(self, feature):
    return preprocessing.normalize(feature)

  def read_feature_file(self, filename):
    with open(filename,'r') as f:
      features = f.readlines()
      features = [d.split('\n')[0] for d in features]
    return features

  def isfloat(self, inp):
    try:
      float(inp)
      return True
    except ValueError:
      return False

  def check_integer(self, number):
    try:
      assert(unicode(str(number),'utf-8').isnumeric())
    except AssertionError:
      print "Expected type {0}, got type {1} for {2}".format(type(number), 'int', number)
      sys.exit(1)

  def check_float(self, number):
    try:
      assert(self.isfloat(number))
    except AssertionError:
      print "Expected type {0}, got type {1} for {2}".format(type(number), 'float', number)
      sys.exit(1)

  def confmat_heatmap(self, cm, score, path):
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('True label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig(path)

  def roc_auc(self, fpr, tpr, auc, path):
    plt.figure(figsize=(9,9))
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve with AUC: {0}'.format(auc), size=15)
    plt.savefig(path)

  def check_dir(self, path):
    if not os.path.exists(path):
      try:
        print "Attempting to create {0} directory".format(path)    
        os.makedirs(path)
        print "{0} directory successfully created".format(path)
      except OSError:
        print "Unable to create {0} directory".format(path)
        sys.exit(1)
