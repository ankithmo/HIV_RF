import sys
sys.path.append("data/.")
sys.path.append("src/.")
sys.path.append("vars/.")
from preproc import preprocess
from split import split
from LR import LR

################################################################################
# PREPROCESSING
################################################################################
# Preprocess?
do_preprocess = True

# Absolute path to the dataset to preprocess
pre_dataset = "data/HIVwidesub2.xls"

# Starting year
start_year = 2011

# Ending year
end_year = 2016

# Absolute path to the resulting dataset
final_dataset = "data/HIV_RF_2016.csv"
################################################################################
# SPLIT DATA
################################################################################
# Split the dataset?
# (data, labels, train, test, unknown)
do_split = True

# Absolute path to the dataset to split
spl_dataset = final_dataset

# Target variable
target = "RF2016"

# Percentage of the dataset must be test
test_size = 0.2

# Absolute directory to the path where the following files must be saved:
# 1. X.csv
# 2. Y.csv
# 3. X_train.csv
# 4. Y_train.csv
# 5. X_test.csv
# 6. Y_test.csv
# 7. X_unknown.csv
# 8. Y_unknown.csv
output_direc = "data"

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
X = "data/X.csv"
Y = "data/Y.csv"
input_list = [X,Y]

# Absolute path to the directory where the results are to be saved
results_path = "results"

# Seed
lr_seed = 123

# Number of folds for cross-validation
k_folds = 10
################################################################################
################################################################################
if __name__ == "__main__":
  if do_preprocess:
    preprocess(pre_dataset, start_year, end_year, final_dataset)
  if do_split:
    split(spl_dataset, target, test_size, output_direc, spl_seed)
  if do_lr:
    logit = LR(input_list, results_path, lr_seed, k_folds)
