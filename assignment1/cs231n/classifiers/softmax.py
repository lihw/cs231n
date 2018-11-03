import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # loss
  num_train = X.shape[0]
  for i in range(num_train):
    f = np.matmul(X[i], W)
    f = np.exp(f) 
    f = f / np.sum(f)
    loss += -np.log(f)[y[i]]
  loss /= num_train
  loss += reg * 0.5 * np.sum(W * W)
  
  # gradient
  f = np.matmul(X, W)
  exp_f = np.exp(f)
  probs = exp_f / np.sum(exp_f, axis = 1, keepdims = True)

  dprobs = probs
  dprobs[range(num_train), y] -= 1
  dprobs /= num_train

  for i in range(num_train):
    dW += np.outer(X[i], dprobs[i])
  
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # loss
  f = np.matmul(X, W)
  exp_f = np.exp(f)
  probs = exp_f / np.sum(exp_f, axis = 1, keepdims = True)
  loss = sum(-np.log(probs[range(num_train), y])) / num_train

  # gradient
  dprobs = probs
  dprobs[range(num_train), y] -= 1
  dprobs /= num_train

  dW = np.matmul(X.T, dprobs)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

