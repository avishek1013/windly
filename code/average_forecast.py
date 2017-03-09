import csv 
import numpy as np

from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



# Note: data has the following form:
# [timestamp,u,v,ws,wd,energy]
def load_data(filepath):
  print "Loading Data\n"
  data = defaultdict(list)

  with open(filepath, 'rb') as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
      windfarm_id = int(row[2])
      try:
        data_row = [int(row[1])] + map(float, row[4:]) + [float(row[3])]
        data[windfarm_id].append(data_row)
      except:
        continue

  for i in data.keys():
    data[i] = np.array(data[i])

  return data

def split_data(data, windfarm_id, ratio):
  print "Splitting Data\n"
  return train_test_split(data[windfarm_id][:,0].reshape(-1, 1), 
    data[windfarm_id][:,-1], test_size = ratio, random_state=0)

# n_estimators=250, learning_rate=1.0, max_depth=2, 
#     min_samples_split = 50, random_state = 0
def run_gbrt(x_train, x_test, y_train, y_test):
  print "Training GBRT\n"
  params = {'n_estimators':[750, 1000, 1200], 'learning_rate':[0.1, 1.0], 
    'max_depth':[1, 2, 3]}
  gbrt = GradientBoostingRegressor(random_state = 0, min_samples_split = 100, max_features = 'sqrt')
  grid = GridSearchCV(gbrt, params, scoring = 'neg_mean_squared_error', n_jobs = 4)
  grid.fit(x_train, y_train)

  print "MSE of Model Prediction"
  print "Training MSE", mean_squared_error(y_train, grid.predict(x_train))
  print "Test MSE", mean_squared_error(y_test, grid.predict(x_test))

  print "MSE of Predicting Average"
  print "Average MSE", mean_squared_error(y_test, [np.mean(y_train)] * len(y_test))
  

def main():
  filepath = "../data/virtual_aggregate_data.csv"
  
  # Load the data and split it
  data = load_data(filepath)
  x_train, x_test, y_train, y_test = split_data(data, 1, 0.3)

  # Train and test using gradient boosted regression trees
  run_gbrt(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
  main()