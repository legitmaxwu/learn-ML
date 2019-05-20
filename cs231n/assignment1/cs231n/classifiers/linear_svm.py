from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            # skip comparing the correct class to itself
            if j == y[i]:
                continue
            # want scores[j] to be significantly less than correct_class_score
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            
            if margin > 0:
                loss += margin
                dW[:,j] += X[i].T
                dW[:,y[i]] -= X[i].T
                
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # reg * (sum(W_ij^2))

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W) # (N, C)
    correct_class_scores = scores[np.arange(num_train),y].reshape(num_train, 1) #(N, 1)
#     print(np.arange(num_train).shape, y.shape)
#     print(correct_class_scores.shape)
#     print(correct_class_scores)
#     print(scores.shape)
#     print(correct_class_scores.shape)
    margin = np.maximum(0, scores - correct_class_scores + 1) # dim: (N, C)
    margin[np.arange(num_train),y] -= 1 
    loss = margin.sum()
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
        updating the gradient
    
        if (class_score + 1 > correct_class_score): (correct class score isn't significantly greater than class score)
            'decrease' class_score by INCREASING dW for that class by x
            'increase' correct_class_score by DECREASING dW for the correct class by x
            (since we update W via W - dW)
         
        this logic, vectorized, translates to:
        
        if (posMargin[i, j] == 1): (i = sample, j = class)
            increment dW[:,j] by X[i].T
            decrement dW[:,y[i]] by X[i].T        
    '''
    
    # add sample x to the gradient-fragment that influences class c if c_score + 1 > correct_class_score
    posMargin = (margin > 0) * 1 # 1 if sample N's margin for class N > 0.
    
    # take away x from the gradient-fragment that infuences the correct class
    takeAwayX = np.zeros(posMargin.shape)
    takeAwayX[np.arange(num_train), y] = np.sum(posMargin, axis=1)
    
    dW += X.T.dot(posMargin)
    dW -= X.T.dot(takeAwayX)
    
    # margin/posMargin is (N, C)
    # X is (N, D)
    # W, dW is (D, C)
    # y is (N, 1)
    
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
