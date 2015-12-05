#!/usr/bin/env python

import numpy as np

def training():
  inputs  = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
  outputs = np.array([[0,0,1,1]]).T

  np.random.seed(1)
  weights_0 = 2*np.random.random((3,4)) - 1
  weights_1 = 2*np.random.random((4,1)) - 1

  for j in xrange(10000):
    layer_0 = inputs

    # forward propagation
    layer_1 = sigmoid(layer_0.dot(weights_0)) # 4x3 dot 3x4 = 4x4
    layer_2 = sigmoid(layer_1.dot(weights_1)) # 4x4 dot 4x1 = 4x1

    layer_2_error = outputs - layer_2 # 4x1 - 4x1 = 4x1
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2) # 4x1 * 4x1 = 4x1

    # backpropagation
    layer_1_error = layer_2_delta.dot(weights_1.T) # 4x1 dot 1x4 = 4x4
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1) # 4x4 * 4x4 = 4x4

    # update weights
    weights_1 += layer_1.T.dot(layer_2_delta)
    weights_0 += layer_0.T.dot(layer_1_delta)

  print "Expected outputs"
  print outputs
  print "Output After Training"
  print layer_2

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
  return np.multiply(x,1-x)

if __name__ == "__main__":
  training()
