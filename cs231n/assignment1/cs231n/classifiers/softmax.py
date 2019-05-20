from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    # print("num_train:", num_train)
    num_classes = W.shape[1]
    # print("num_classes:", num_classes)
    
    for i in range(num_train):
        scores = X[i].dot(W) # scores is 1 * C
        correct_class = y[i]
        
        # LOSS DUE TO TRAINING SAMPLE = -log(exp^correct_score / sum(exp^all_other_scores))
        log_c = np.max(scores)
        scores -= log_c
        correct_class_score = scores[correct_class]
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(np.exp(scores))
        proportion = np.exp(correct_class_score) / sum_exp_scores
        loss -= np.log(proportion)
        # print(proportion)
        
        # ALTERNATIVELY: (we split the log)
#         loss -= scores[y[i]]
#         loss += np.log(np.sum(np.exp(X[i].dot(W))))
        
        # UPDATE GRADIENT
        for j in range(num_classes):
            p = np.exp(scores[j]) / sum_exp_scores # "probability" of class j
            dW[:,j] += (p - (j == y[i])) * X[i,:]
            # dW is D by C

    loss /= num_train
    loss += reg * np.sum(W * W) 
    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    # print("num_train:", num_train)
    num_classes = W.shape[1]
    # print("num_classes:", num_classes)
    
    scores = X.dot(W) # scores is N*D x D*C -> N*C    
    log_c = np.max(scores, axis=1).T
    scores -= log_c[:,None]
    correct_class_score = scores[np.arange(num_train),y]
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(np.exp(scores), axis=1)
    proportion = np.exp(correct_class_score) / sum_exp_scores
    loss -= np.sum(np.log(proportion))
    
    # calculating dW = (p - (c = correct c ? 1 : 0)) * x
    correct_class_one_hot = np.zeros_like(scores)
    correct_class_one_hot[np.arange(num_train),y] += 1
    p = np.exp(scores) / sum_exp_scores[:,None] - correct_class_one_hot # N*C / N:1 -> N*C
    dW += X.T.dot(p) # D*N x N*C -> D*C

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W) 
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
