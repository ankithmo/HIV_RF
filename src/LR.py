import sys
sys.path.append("../.")

from functions import Helpers
from variables import Variables
from setup_logger import logger

def LR(input_list, results_path, seed=123, k_folds=10):
  """
    1. Perform k-fold logistic regression on X and Y
    2. Get heatmap of confusion matrix
    3. Get ROC curve

    Arguments:
      - input_list: list, length = 2 or 4
          Absolute path to [X,Y] or [X_train, Y_train, X_test, Y_test]
      - results_path: str
          Absolute path to the directory where the figures must be saved
      - seed: int, optional, default = 123
          Random seed
      - k_folds: int, optional, default = 10
          Number of folds for cross-validation

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
    logger.error("{0} files found in input_list, expected 2 or 4".format(num_files))
    h.error()

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
    logger.info("accuracy = {0}".format(score))

    # build confusion matrix
    cm = confusion_matrix(Y, lr.predict(X))
    h.confmat_heatmap(cm, score, os.path.join(results_path,'confmat.png'))

    # build roc auc curve
    fpr, tpr, _ = roc_curve(lr.predict(X), Y, drop_intermediate=False)
    auc = roc_auc_score(lr.predict(X), Y)
    logger.info("ROC AUC = {0}".format(auc))
    h.roc_auc(fpr, tpr, auc, os.path.join(results_path,'roc_auc.png'))
  else:
    lr = LogisticRegression(solver='liblinear', random_state=seed)
    lr.fit(X_train, Y_train)

    # get accuracy, confusion matrix and ROC AUC
    h.get_metrics(lr, [X_train, Y_train, X_test, Y_test], results_path)

  return lr
