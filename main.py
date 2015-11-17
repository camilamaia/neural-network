#!/usr/bin/env python

import csv
import numpy as np
from numpy import matrix

training_set       = []
generalization_set = []
validation_set     = []


def read_data():
  with open('data_sets/letter-a-recognition.csv', 'rb') as f:
    global training_set, generalization_set, validation_set
    lines              = list(csv.reader(f))
    file_size          = len(lines)
    training_set       = lines[:int(file_size*0.6)]
    generalization_set = lines[int(file_size*0.6):int(file_size*0.8)]
    validation_set     = lines[int(file_size*0.8):]


def training():
  global training_set
  number_of_rows     = len(training_set)
  number_of_features = 16
  training_set       = training_set[:number_of_rows]

  inputs  = np.delete(matrix(training_set),-1,1).astype('int32') # (5x16)
  outputs = np.delete(matrix(training_set),range(16),1).astype('int32') # (5x1)

  np.random.seed(1)
  weights0 = 2*np.random.random((number_of_features,1)) - 1 # (16x1)

  for j in xrange(10000):
    # forward propagation
    l1 = nonlin(np.dot(inputs,weights0)) #5x1

    # how much did we miss?
    l1_error = outputs - l1 #5x1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = np.multiply(l1_error,nonlin(l1,True))

    # update weights
    weights0 += np.dot(inputs.T,l1_delta)

  print "Error After Training:"
  l1 = np.matrix.round(l1)
  error = np.matrix.mean(outputs - l1)*100


# sigmoid function
def nonlin(x,deriv=False):
  if(deriv==True):
    return np.multiply(x,1-x)
  return 1/(1+np.exp(-x))


if __name__ == "__main__":
  read_data()
  training()
