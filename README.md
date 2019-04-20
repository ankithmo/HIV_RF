# To predict kidney disease among individuals who have HIV. 

## Prerequisites

* python v3.7.3 must be installed
* The following packages must be installed:
  * pandas v0.24.2
  * numpy v1.16.2
  * logging 
  * os
  * matplotlib
      * pyplot v3.0.3
  * seaborn v0.9.0
  * xlrd v0.9.0
  * pydotplus v2.0.2
  * sklearn v0.20.3
      * model_selection
      * preprocessing
      * tree
      * neural_network
      * externals.six
      * linear_model
      * metrics
  * sys

## Execution

* Update parameters in `frontend.py`
* Update the names of the diseases in `diseases.txt` under `vars` directory
* Update the names of the features in `features.txt` under `vars` directory
* Execute `python3 frontend.py` from `HIV_RF` directory
