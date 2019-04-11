import numpy as np
import pandas as pd
import os
import sys
import xlrd
import string

def preprocess(dataset, start_year , end_year, final_dataset, diseases_f="../diseases.txt"):
  """
    Does the following:
      1. Imports dataset
      2. Selects only PLWHA without RF in start_year
      3. For each disease, if disease in any year from start_year to end_year-1, then disease present in end_year

    Arguments:
      - dataset: Absolute path to the dataset xls file
      - start_year: Starting year
      - end_year: Ending year
      - final_dataset: Absolute path to the xls file where the preprocessed data must be written to

    Returns:
      - No return values
  """
  # Check
  try:
    assert(os.path.exists(dataset))
  except AssertionError:
    print "{0} does not exist".format(dataset)

  try:
    assert(end_year > start_year)
  except AssertionError:
    print "end year must be greater than start year"

  try:
    assert(final_dataset.split('.')[-1] == 'xls')
  except AssertionError:
    print "target file must be a .xls file"

  # Import dataset
  print "Reading {0}".format(dataset)
  if dataset.split('.')[1] is 'csv':
    data = pd.read_csv(dataset)
  else:
    data = pd.read_excel(dataset)
  print "{0} has {1} rows and {2} columns".format(dataset, data.shape[0], data.shape[1])

  # Individuals in data who have HIV at start_year 
  data_hiv = data[data["HIV"+str(start_year)]==1]

  # Individuals in data_HIV who don't have RF at start_year
  data_hiv_no_rf = data_hiv[data_hiv["RF"+str(start_year)]==0]
  print "{0} PLWHA without RF in {1}".format(data_hiv_no_rf.shape[0], start_year)

  # Read diseases from disease file
  with open(diseases_f,'r') as f:
    diseases = f.readlines()
  diseases = [d.split('\n')[0] for d in diseases]
  
  # Make a copy of data_hiv_no_rf
  data_hiv_no_rf_copy = data_hiv_no_rf.copy()

  print "Processing disease values of {0} based on previous years".format(end_year)
  # For each disease, set value 1 at end_year if value is 1 at any previous year
  for year in range(start_year, end_year):
    for disease in diseases:
      data_hiv_no_rf_copy.ix[data_hiv_no_rf_copy[disease+str(year)]==1, [disease+str(end_year)]] = 1

  # Save data_hiv_no_rf_copy to final_dataset
  data_hiv_no_rf_copy.to_csv(final_dataset, index = False)
  print "{0} generated successfully".format(final_dataset)

  return None

  

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print "Input invalid"
        print "python preproc.py <input path> <start year> <end year> <target path>"
        sys.exit()

    if not unicode(sys.argv[2],'utf-8').isnumeric():
        print "start_year should be an integer"
        sys.exit()

    if not unicode(sys.argv[3],'utf-8').isnumeric():
        print "end_year should be an integer"
        sys.exit()

    preprocess(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
