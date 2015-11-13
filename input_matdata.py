__author__ = 'xinwen'
import numpy
import os
import scipy.io as sio
import matplotlib.pyplot as plt

data=sio.loadmat('MNIST_data/ex3data1.mat')
X=data['X']
y=data['y']
