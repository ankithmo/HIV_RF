import numpy as np
import pandas as pd
import os
import sys
import xlrd
sys.path.append("../.")

from functions import Helpers
from variables import Variables

def preprocess(dataset, start_year, end_year, final_dataset):
  """
    Does the following:
      1. Imports dataset
      2. Selects only PLWHA without RF in start_year
      3. For each disease, if disease in any year from start_year to end_year-1, then disease present in end_year

    Arguments:
      - dataset: Absolute path to the dataset xls file
      - start_year: Starting year
      - end_year: Ending year
      - final_dataset: Absolute path to the file where the preprocessed data must be written to

    Returns:
      - No return values
  """
  h = Helpers()
  v = Variables()

  # Diagnostics
  h.check_file_existence(dataset)
  h.check_year(start_year, end_year)
  h.check_dir(os.path.dirname(final_dataset))

  # Import dataset
  data = h.import_dataset(dataset)

  # Individuals in data who have HIV at start_year 
  data_hiv = data[data["HIV"+str(start_year)]==1]

  # Individuals in data_HIV who don't have RF at start_year
  data_hiv_no_rf = data_hiv[data_hiv["RF"+str(start_year)]==0]
  print "{0} PLWHA without RF in {1}".format(data_hiv_no_rf.shape[0], start_year)

  # Read diseases from disease file
  diseases = h.read_feature_file(v.diseases_f)
  
  # Make a copy of data_hiv_no_rf
  data_hiv_no_rf_copy = data_hiv_no_rf.copy()

  print "Processing disease values of {0} based on previous years".format(end_year)
  # For each disease, set value 1 at end_year if value is 1 at any previous year
  for year in range(start_year, end_year):
    for disease in diseases:
      data_hiv_no_rf_copy.ix[data_hiv_no_rf_copy[disease+str(year)]==1, [disease+str(end_year)]] = 1

  # Save data_hiv_no_rf_copy to final_dataset
  h.write_dataset(data_hiv_no_rf_copy, final_dataset)

  return None
