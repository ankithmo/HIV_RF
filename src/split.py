from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys
sys.path.append("../.")

from functions import Helpers
from variables import Variables
from setup_logger import logger

def split(dataset, target, test_size, output_direc, seed=123):
  """
    1. Import dataset
    2. Take only individuals where target is not NaN in dataset
    3. X: Data for required features
    4. Y: Data for target
    5. Split X and Y into train and test
    6. X_unknown and Y_unknown where Y is missing
    7. Save X, Y, X_train, Y_train, X_test, Y_test, X_unknown, Y_unknown to output_direc

    Arguments:
      - dataset: str
          Absolute path to the preprocessed dataset which has to be split
      - target: str
          The target feature
      - test_size: float
          Size of the test set (percentage)
      - output_direc: str
          Directory where training and test directory must be placed
      - seed: int, optional, default = 123
          Random seed value

    Returns:
      - No return value
  """
  h = Helpers()
  v = Variables()

  # Diagnostics
  h.check_file_existence(dataset)
  h.check_float(test_size)
  h.check_file_existence(output_direc)
  h.check_integer(seed)
  h.check_dir(output_direc)

  # Import dataset
  data = h.import_dataset(dataset)

  # Take only those individuals where target is not NaN
  data_nonNaN = data[data[target].notnull()]
  logger.info("{0} individuals do not have missing values in their target attribute".format(data_nonNaN.shape[0]))
  data_NaN = data[data[target].isnull()]

  # Read features from features file
  features = h.read_feature_file(v.features_f)

  # Get X and Y
  X = data_nonNaN[features]
  Y = data_nonNaN[target]

  # Split X and Y into training and testing data
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
  
  # Feature scaling
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  # Get X_unknown and Y_unknown
  X_unknown = data_NaN[features]
  Y_unknown = data_NaN[target]

  # Convert these np arrays to pd dataframes
  X = pd.DataFrame(X, columns=features)
  Y = pd.DataFrame(Y, columns=[target])
  X_train = pd.DataFrame(X_train, columns=features)
  Y_train = pd.DataFrame(Y_train, columns=[target])
  X_test = pd.DataFrame(X_test, columns=features)
  Y_test = pd.DataFrame(Y_test, columns=[target])
  X_unknown = pd.DataFrame(X_unknown, columns=features)
  Y_unknown = pd.DataFrame(Y_unknown, columns=[target])

  # Save train, test and unknown to output_direc
  h.write_dataset(X, os.path.join(output_direc, 'X.csv'))
  h.write_dataset(Y, os.path.join(output_direc, 'Y.csv'))
  h.write_dataset(X_train, os.path.join(output_direc, 'X_train.csv'))
  h.write_dataset(Y_train, os.path.join(output_direc, 'Y_train.csv'))
  h.write_dataset(X_test, os.path.join(output_direc, 'X_test.csv'))
  h.write_dataset(Y_test, os.path.join(output_direc, 'Y_test.csv'))
  h.write_dataset(X_unknown, os.path.join(output_direc, 'X_unknown.csv'))
  h.write_dataset(Y_unknown, os.path.join(output_direc, 'Y_unknown.csv'))

  return None
