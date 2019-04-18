import sys
sys.path.append("../.")

from functions import Helpers
from variables import Variables
from setup_logger import logger

def NN(input_list, results_path, seed=123, hidden_layer_sizes=(30,30,30), activation='relu', solver='adam', regularization=0.0001, batch_size='auto', learning_rate_sch='constant', learning_rate_init=0.001, max_iter=100, tol=1e-4, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, n_iter_no_change=10):
  """
    1. Train a neural network to classify X based on Y
    2. Get classification report
    3. Get heatmap of confusion matrix
    4. Get ROC curve

    Arguments:
      - input_list : list, length = 4
          Absolute path to [X_train, Y_train, X_test, Y_test]
      - results_path: str 
          Absolute path to the directory where the files will be saved
      - seed: int, optional, default = 123
          Random seed
      - hidden_layer_sizes: tuple, length = n_layers-2, default = (30,30,30)
          Number of neurons in each layer
      - activation: {'identity', 'logistic', 'tanh', 'relu'}, optional, default = 'relu'
          Activation function
      - solver: {'lbfgs', 'sgd', 'adam'}, optional, default = 'adam'
          Solver
      - regularization: float, optional, default = 0.0001
          L2 regularization term
      - batch_size: int, optional, default = 'auto'
          Minibatch size
      - learning_rate_sch: {'constant', 'invscaling', 'adaptive'}, optional, default='constant'
          Learning rate schedules
      - learning_rate_init: float, optional, default = 0.001
          Initial learning rate used
      - max_iter: int, optional, default = 100
          Maximum number of iterations
      - tol: float, optional, default = 1e-4
          Tolerance for the optimization
      - momentum: float, optional, default = 0.9
          Momentum for gradient descent update
      - nesterovs_momentum: boolean, optional, default = True
          Whether to use Nesterov's momentum
      - early_stopping: bool, optional, default = False
          Whether to use early stopping to terminate training when validation score is not improving
      - validation_fraction: float, optional, default = 0.1
          Proportion of training data to set aside as validation set for early stopping
      - beta_1: float in [0,1), optional, default = 0.9
          Exponential decay rate for estimates of first moment vector in adam
      - beta_2: float in [0,1), optional, default = 0.999
          Exponential decay rate for estimates of second moment vector in adam
      - n_iter_no_change: int, optional, default = 10
          Maximum number of epochs to not meet tol improvement

    Returns:
      - Trained neural network
  """
  h = Helpers()
  v = Variables()

  # Diagnostics
  h.check_dir(results_path)

  if len(input_list) != 4:
    logging.error("{0} files found in input_list, expected 4".format(len(input_list)))
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

  # Train NN
  nn = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, 
          activation=activation, 
          solver=solver, 
          alpha=regularization, 
          batch_size=batch_size, 
          learning_rate=learning_rate_sch, 
          learning_rate_init=learning_rate_init, 
          max_iter=max_iter, 
          random_state=seed, 
          tol=tol, 
          momentum=momentum, 
          nesterovs_momentum=nesterovs_momentum, 
          early_stopping=early_stopping, 
          validation_fraction=validation_fraction, 
          beta_1=beta_1, 
          beta_2=beta_2, 
          n_iter_no_change=n_iter_no_change)
  nn.fit(X_train, Y_train)

  # get accuracy, confusion matrix and ROC AUC
  h.get_metrics(nn, [X_train, Y_train, X_test, Y_test], results_path)

  return nn
