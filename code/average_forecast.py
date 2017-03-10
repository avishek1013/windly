import csv 
import numpy as np

from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



# Note: data has the following form:
# [timestamp,u,v,ws,wd,energy]
def load_data(filepath):
  print "Loading Data"
  data = defaultdict(list)
  historical = defaultdict(list)
  gap = 1
  hours = 12

  with open(filepath, 'rb') as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
      wid = int(row[2])
      try:
        data_row = [int(row[1])] + map(float, row[4:]) + [float(row[3])]

        if len(historical[wid]) <= gap:
          data_row = data_row[:-1] + [0.0]*hours + [data_row[-1]]
        else:
          fill = gap + hours - len(historical[wid])
          data_row = data_row[:-1] + \
            [0.0]*fill + \
            historical[wid][:(hours - fill)] + \
            [data_row[-1]]

        if len(historical[wid]) >= gap + hours:
          historical[wid].pop(0)

        historical[wid].append(float(row[3]))
        data[wid].append(data_row)
      except:
        continue

  for i in data.keys():
    data[i] = np.array(data[i])

  return data

def split_data(data, windfarm_id, ratio):
  print "Splitting Data"
  return train_test_split(data[windfarm_id][:,:-1], 
    data[windfarm_id][:,-1], test_size = ratio, random_state=0)

def split_data_cont(data, windfarm_id, ratio):
  print "Splitting Data Contiguous"
  n = len(data[windfarm_id])
  b = int(ratio*n)
  x_train = data[windfarm_id][:b, 1:-1]
  y_train = data[windfarm_id][:b, -1]
  x_test = data[windfarm_id][b:, 1:-1]
  y_test = data[windfarm_id][b:, -1]
  return x_train, x_test, y_train, y_test


def run_gbrt(x_train, x_test, y_train, y_test):
  print "\nTraining GBRT"

  params = {'n_estimators':[700], 'learning_rate':[0.1], 
    'max_depth':[2]}
  gbrt = GradientBoostingRegressor(random_state = 0, min_samples_split = 100, max_features = 'sqrt')
  grid = GridSearchCV(gbrt, params, scoring = 'neg_mean_squared_error', n_jobs = 4)
  grid.fit(x_train, y_train)

  print "MSE of Model Prediction"
  print "Training MSE", mean_squared_error(y_train, grid.predict(x_train))
  print "Test MSE", mean_squared_error(y_test, grid.predict(x_test))

  print "MSE of Predicting Average"
  print "Average MSE", mean_squared_error(y_test, [np.mean(y_train)] * len(y_test))

  print grid.best_params_

def run_mlp(x_train, x_test, y_train, y_test):
  print "\nTraining MLP"
  params = {'hidden_layer_sizes':[(100,), (20, 20,)], 'learning_rate_init':[0.1, 0.05], 'alpha':[0.001, 0.01]}
  mlp = MLPRegressor(random_state = 0, learning_rate = 'adaptive')
  grid = GridSearchCV(mlp, params, scoring = 'neg_mean_squared_error', n_jobs = 4)
  grid.fit(x_train, y_train)

  print "MSE of Model Prediction"
  print "Training MSE", mean_squared_error(y_train, grid.predict(x_train))
  print "Test MSE", mean_squared_error(y_test, grid.predict(x_test))

  print "MSE of Predicting Average"
  print "Average MSE", mean_squared_error(y_test, [np.mean(y_train)] * len(y_test))

  print grid.best_params_
  

def main():
  filepath = "../data/virtual_aggregate_data.csv"
  
  # Load the data and split it
  data = load_data(filepath)
  x_train, x_test, y_train, y_test = split_data_cont(data, 1, 0.3)

  # Train and test using gradient boosted regression trees
  run_gbrt(x_train, x_test, y_train, y_test)
  # run_mlp(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
  main()