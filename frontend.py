import sys
sys.path.append("data/.")
sys.path.append("src/.")
sys.path.append("vars/.")
from functions import Helpers
from preproc import preprocess
from split import split
from LR import LR
from DT import DT
from NN import NN
from setup_logger import logger
################################################################################
# PREPROCESSING
################################################################################
# Preprocess?
do_preprocess = True

# Absolute path to the dataset to preprocess
pre_dataset = "data/HIVwidesub2.xls"

# Starting year
pre_start_year = 2011

# Ending year
pre_end_year = 2016

# Absolute path to the resulting dataset
pre_final_dataset = "data/HIV_RF_2016.csv"
################################################################################
# SPLIT DATA
################################################################################
# Split the dataset?
# (data, labels, train, test, unknown)
do_split = True

# Absolute path to the dataset to split
spl_dataset = pre_final_dataset

# Target variable
spl_target = "RF2016"

# Percentage of the dataset must be test
spl_test_size = 0.2

# Absolute directory to the path where the following files must be saved:
# 1. X.csv
# 2. Y.csv
# 3. X_train.csv
# 4. Y_train.csv
# 5. X_test.csv
# 6. Y_test.csv
# 7. X_unknown.csv
# 8. Y_unknown.csv
spl_output_direc = "data"

# Seed value
spl_seed = 123
################################################################################
# LOGISTIC REGRESSION
################################################################################
# Perform logistic regression?
do_lr = True

# Input list of absolute paths
# input_list = [<path_to_datapoints>,<path_to_labels>]
# or
# input_list = [<path_to_X_train>,<path_to_Y_train>,<path_to_X_test>,<path_to_Y_test>]
lr_X = "data/X.csv"
lr_Y = "data/Y.csv"
lr_input_list = [lr_X,lr_Y]

# Absolute path to the directory where the results are to be saved
lr_results_path = "results/lr"

# Seed
lr_seed = 123

# Number of folds for cross-validation
lr_k_folds = 10
################################################################################
# DECISION TREE
################################################################################
# Perform decision tree classification?
do_dt = True

# Input list of absolute paths
# input_list = [<path_to_X_train>,<path_to_Y_train>,<path_to_X_test>,<path_to_Y_test>
dt_X_train = "data/X_train.csv"
dt_Y_train = "data/Y_train.csv"
dt_X_test = "data/X_test.csv"
dt_Y_test = "data/Y_test.csv"
dt_input_list = [dt_X_train,dt_Y_train,dt_X_test,dt_Y_test]

# Absolute path to the directory where the results are to be saved
dt_results_path = "results/dt"

# Seed
dt_seed = 123

# Criterion to measure the quality of the split
# 'gini' or 'entropy'
dt_criterion = 'entropy'

# Strategy used to choose the split at each node
# 'best' or 'random'
dt_splitter = 'best'

# Maximum depth of tree
# 'None' or int value less than depth of tree
dt_max_depth = 5
################################################################################
# NEURAL NETWORK
################################################################################
# Perform neural network classification
do_nn = True

# Input list of absolute paths
# input_list = [<path_to_X_train>,<path_to_Y_train>,<path_to_X_test>,<path_to_Y_test>]
nn_X_train = "data/X_train.csv"
nn_Y_train = "data/Y_train.csv"
nn_X_test = "data/X_test.csv"
nn_Y_test = "data/Y_test.csv"
nn_input_list = [nn_X_train,nn_Y_train,nn_X_test,nn_Y_test]

# Absolute path to the directory where the results are to be saved
nn_results_path = "results/nn"

# Seed
nn_seed = 123

# Size of hidden layer
# number of elements of tuple indicates the number of hidden layers
# i-th entry of tuple indicates number of hidden units in the i-th hidden layer
nn_hidden_layer = (100,)

# Activation function
nn_activation = 'identity'

# Solver
nn_solver = 'adam'

# Regularization parameter
nn_regularization = 0.0001

# Size of minibatches
nn_batch_size = 'auto'

# Learning rate schedule
nn_learning_rate_sch = 'constant'

# Initial learning rate
nn_learning_rate_init = 0.001

# Maximum number of iterations
nn_max_iter = 100

# Tolerance for the optimization
nn_tol = 1e-4

# Momentum
nn_momentum = 0.9

# Nesterov's momentum
nn_nesterov = True

# Whether to use early stopping
nn_early = False

# Proportion of training data to set aside as validation set
nn_valid_frac = 0.1

# Exponential decay rate for estimates of first moment vector in adam
nn_beta_1 = 0.9

# Exponential decay rate for estimates of second moment vector in adam
nn_beta_2 = 0.999

# Maximum number of epochs to not meet tol improvement
nn_iter_no_change = 10
################################################################################
# DO NOT CHANGE CODE
################################################################################
if __name__ == "__main__":
  h = Helpers()
  
  if do_preprocess:
    print("Preprocessing")
    preprocess(pre_dataset, pre_start_year, pre_end_year, pre_final_dataset)
    print("completed")

  if do_split:
    print("\nSplitting dataset")
    split(spl_dataset, spl_target, spl_test_size, spl_output_direc, spl_seed)
    print("completed")

  try:
    if do_lr:
      print("\nLogistic regression")
      logit = LR(lr_input_list, lr_results_path, lr_seed, lr_k_folds)
      print("completed")
  except Exception as e:
    logger.error(e)

  try:
    if do_dt:
      print("\nDecision tree classification")
      dec_tree = DT(dt_input_list, dt_results_path, dt_seed, dt_criterion, dt_splitter, dt_max_depth)
      print("completed")
  except Exception as e:
    logger.error(e)

  try:  
    if do_nn:
      print("\nClassification using neural network")
      neural_net = NN(nn_input_list, nn_results_path, nn_seed, nn_hidden_layer, nn_activation, nn_solver, nn_regularization, nn_batch_size, nn_learning_rate_sch, nn_learning_rate_init, nn_max_iter, nn_tol, nn_momentum, nn_nesterov, nn_early, nn_valid_frac, nn_beta_1, nn_beta_2, nn_iter_no_change)
      print("completed")
  except Exception as e:
    logger.error(e)

  h.done()
