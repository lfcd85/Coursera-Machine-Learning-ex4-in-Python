from __future__ import division
from scipy.special import expit
import numpy as np

## SIGMOIDGRADIENT returns the gradient of the sigmoid function
def sigmoid_gradient(z):
    g = np.zeros(z.shape)
    g = expit(z) * (1 - expit(z))
    return g

## NNCOSTFUNCTION Implements the neural network cost function for a two layer
## neural network which performs classification
def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, param_lambda):
    # initialize variables and constants
    m = np.size(X, 0)
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1))     
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    # Feedforwarding for 3 layer NN
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)    
    z2 = np.dot(a1, Theta1.T)
    a2 = np.concatenate((np.ones((m, 1)), expit(z2)), axis=1)
    z3 = np.dot(a2, Theta2.T)
    a3 = expit(z3)
    
    # Calculate cost function J
    for k in range(1, num_labels+1):
        y_digit = (y == k) * 1
        J = J + ((1 / m) * ((y_digit.T * -1) * np.log(a3[:,k-1]) - (1 + (y_digit.T * -1)) * np.log(1 - a3[:,k-1]))).sum()
    Theta1_cost = (param_lambda / (2 * m)) * Theta1 ** 2 
    Theta2_cost = (param_lambda / (2 * m)) * Theta2 ** 2     
    J = J + Theta1_cost[:, 1:np.size(Theta1_cost, 1)].sum() + Theta2_cost[:, 1:np.size(Theta2_cost, 1)].sum()
    
    # Backpropagation for 3 layer NN
    delta3 = np.zeros(a3.shape)
    for k in range(1, num_labels+1):
        y_digit = (y == k) * 1
        delta3[:,k-1] = a3[:,k-1] - y_digit.T    
    delta2 = np.dot(delta3, Theta2[:, 1:]) * sigmoid_gradient(z2) 
    DELTA2 = np.dot(delta3.T, a2) 
    DELTA1 = np.dot(delta2.T, a1)
    
    # Regularize gradients
    Theta1_grad = (1 / m) * DELTA1 
    Theta2_grad = (1 / m) * DELTA2 
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (param_lambda / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (param_lambda / m) * Theta2[:, 1:]
    
    # Unroll gradients
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size), Theta2_grad.reshape(Theta2_grad.size)))
    return [J, grad]

## RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
## incoming connections and L_out outgoing connections
def rand_initialize_weights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    W = np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    return W

# PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)
def predict(Theta1, Theta2, X):
    m = np.size(X, 0)
    p = np.zeros(m)
    
    h1 = expit(np.dot(np.concatenate((np.ones((m, 1)), X), axis=1) , Theta1.T))
    h2 = expit(np.dot(np.concatenate((np.ones((m, 1)), h1), axis=1) , Theta2.T))
    p = np.argmax(h2, axis=1) + 1
    return p
