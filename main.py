#!/usr/bin/env python

import csv

def read_data():
  with open('data_sets/letter-a-recognition.csv', 'rb') as f:
    lines           = list(csv.reader(f))
    file_size       = len(lines)
    training        = lines[:int(file_size*0.6)]
    generalization  = lines[int(file_size*0.6*1):int(file_size*0.8)]
    validation      = lines[int(file_size*0.8*1):]

if __name__ == "__main__":
  read_data()
