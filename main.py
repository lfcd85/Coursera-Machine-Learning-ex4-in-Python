## Initialization
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as op
import os

from modules import nn_cost_function, rand_initialize_weights, predict

# Setup the parameters
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)
print 'Input Layer Size: ' + str(input_layer_size)
print 'Hidden Layer Size: ' + str(hidden_layer_size)
print 'Number of Labels: ' + str(num_labels) + '\n'

# Load data
print 'Loading and Shuffling Data ...\n'
data_filepass = os.path.abspath(os.path.dirname(__file__)) + '/ex4data1.mat'
data1 = sio.loadmat(data_filepass)

# Shuffle data
Xy = np.concatenate((data1['X'], data1['y']), axis=1)
Xy_shuffle = np.zeros(Xy.shape)
for i in range(np.size(Xy, 0)):
    Xy_shuffle[i] = Xy[np.random.permutation(range(np.size(Xy, 0)))[i]]

X_shuffle = Xy_shuffle[:, :400]
y_shuffle = Xy_shuffle[:, 400]

# Divide data into training set and cross validation set
div_point = 4500
X = X_shuffle[:div_point]
X_cval = X_shuffle[div_point:]
y = y_shuffle[:div_point]
y_cval = y_shuffle[div_point:]
m = np.size(X, 0)

# Randomly select 100 data points to display
sel = np.random.permutation(range(m))[0:100]
img = np.zeros([200, 200])
for i in range(sel.size):
    column = int(i / 10)
    row = int(i % 10)
    img[(20*(column)):(20*(column+1)), (20*(row)):(20*(row+1))] = (X[sel[i]].reshape(20, 20)).T

print 'Visualizing Some Data Examples (Check Them and then Close the Window)\n'
imgplot = plt.imshow(img, interpolation='nearest')
plt.gray()
plt.show()

print 'Initializing Neural Network Parameters ...\n'

initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.reshape(initial_Theta1.size), initial_Theta2.reshape(initial_Theta2.size)])

# set the values of lambda and iteration
param_lambda = 0.03
max_iter = 100
print 'The Value of Lambda: ' + str(param_lambda)
print 'Iteration Times: ' + str(max_iter) + '\n'

# minimize the cost_function
print 'Training Neural Network ...'
result_nn_params = op.minimize(fun = nn_cost_function, x0 = initial_nn_params, method = 'CG', jac = True,
                               options = {'maxiter':max_iter, 'disp':True},
                               args = (input_layer_size, hidden_layer_size, num_labels, X, y, param_lambda))

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = result_nn_params.x[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
Theta2 = result_nn_params.x[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1)) 

# Prediction and its accuracy for training set
pred = predict(Theta1, Theta2, X)
accuracy = np.mean(((pred == y.T)*1) * 100)
print '\nTraining Set Accuracy: ' + str(accuracy) + '\n'

# Prediction and its accuracy for cross validation set
pred_cval = predict(Theta1, Theta2, X_cval)
accuracy_cval = np.mean(((pred_cval == y_cval.T)*1) * 100)
print 'Cross Validation Set Accuracy: ' + str(accuracy_cval) + '\n'

