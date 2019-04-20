from sklearn.tree import DecisionTreeClassifier
import os
import sys
sys.path.append("../.")

from functions import Helpers
from variables import Variables
from setup_logger import logger

def DT(input_list, results_path, seed=123, criterion='gini', splitter='best', max_depth=None):
  """
    1. Perform pruned decision tree classification on X and Y
    2. Get heatmap of confusion matrix
    3. Get decision tree
    4. Get ROC curve

    Arguments:
      - input_list: list, length = 4
          Absolute path to [X_train, Y_train, X_test, Y_test]
      - results_path: str
          Absolute path to the directory where the files will be saved
      - seed: int, optional, default = 123
          Random seed
      - criterion: {gini, entropy}, optional, default = gini
          Function to measure the quality of a split
      - splitter: {best, random}, optional, default = best
          Strategy used to choose the split at each node
      - max_depth: int, value <= length of tree, optional, default = None
          Maximum depth of the tree
     
    Returns:
      - Trained pruned decision tree classifier
  """
  h = Helpers()
  v = Variables()

  # Diagnostics
  h.check_dir(results_path)

  if len(input_list) != 4:
    logger.error("{0} files found in input_list, expected 4".format(len(input_list)))
    h.error()
    
  X_train, Y_train, X_test, Y_test = input_list
  h.check_file_existence(X_train)
  h.check_file_existence(Y_train)
  h.check_file_existence(X_test)
  h.check_file_existence(Y_test)

  # Import datasets
  X_train = h.import_dataset(X_train)
  Y_train = h.import_dataset(Y_train)
  X_test = h.import_dataset(X_test)
  Y_test = h.import_dataset(Y_test)

  # Train DT
  dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=seed)
  dt.fit(X_train, Y_train)

  # get accuracy, confusion matrix and ROC AUC
  h.get_metrics(dt, [X_train, Y_train, X_test, Y_test], results_path)

  # build decision tree
  h.decision_tree_viz(dt, os.path.join(results_path,'decision_tree.png'))

  return dt
