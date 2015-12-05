 #!/usr/bin/env python

import numpy as np

def training():
  inputs  = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1] ])
  outputs = np.array([[0,0,1,1]]).T

  np.random.seed(1)
  weights_0 = 2*np.random.random((3,1)) - 1

  for j in xrange(10000):
    layer_0 = inputs

    # forward propagation
    layer_1 = sigmoid(np.dot(layer_0, weights_0))

    # how much did we miss?
    layer_1_error = outputs - layer_1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in layer_1
    layer_1_delta = np.multiply(layer_1_error, sigmoid_derivative(layer_1))

    # update weights
    weights_0 += np.dot(layer_0.T,layer_1_delta)

  print "Expected outputs"
  print outputs
  print "Output After Training"
  print np.round(layer_1)

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
  return np.multiply(x,1-x)

if __name__ == "__main__":
  training()
