function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

identity = eye(num_labels);
Y = identity(y,:);

X = [ones(m,1) X];
z2 = X * Theta1';
a2 = sigmoid(z2); % a is now m by layer features (i.e. 5000 by 25)
z3 = [ones(m,1) a2] * Theta2';
a3 = sigmoid(z3); % h is now m by output features (i.e.  5000 by 10)
t1 = log(a3) .* Y; % Y is m by number of labels
t2 = log(1 - a3) .* (1 - Y);
J = J - 1/m * (sum(sum(t1 + t2)));
J = J + lambda/2/m * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

for t = 1:m
  % a1 = X(t,:); % a1 is 1 by (input features + 1 bias)
  % z2 = a1 * Theta1'; % z2 is 1 by  (input features + 1 bias)
  % a2 = [1 sigmoid(z2)]; % a2 is 1 by (a2 features + 1 bias)
  % z3 = a2 * Theta2'; % z3 is 1 by output features
  % a3 = sigmoid(z3); % a3 is 1 by output features
  
  % d3 = a3(t,:) .- identity(y(t),:); % d3 is 1 by output features
  % d2 = (d3 * Theta2)(:,2:end) .* sigmoidGradient(z2(t,:)); % d2 is 1 by a2 features (we ignore delta-0)
  % Theta1(2:end,:) = Theta1(2:end,:) + d2(:,2:end)' * X(t,:); % Theta1 equals input features by a2 features
  % Theta2 = Theta2 + d3' * [1 a2(t,:)]; % Theta2 equals a2 features by output features
  
  % a1 = X(t,:), a2 = [1 a2(t,:)], a3 = a3(t,:), z2 = z2(t,:), z3 = z3(t,:)
  d3 = a3(t,:)' .- identity(y(t),:)';
  trash_value = 404;
  d2 = (Theta2' * d3) .* [trash_value; sigmoidGradient(z2(t,:)')];
  d2 = d2(2:end);
  Theta1_grad = Theta1_grad + d2 * X(t,:);
  Theta2_grad = Theta2_grad + d3 * [1 a2(t,:)];
endfor

Theta1_grad = Theta1_grad ./ m;
Theta1_grad(:,2:end) += lambda / m .* Theta1(:,2:end);
Theta2_grad = Theta2_grad ./ m;
Theta2_grad(:,2:end) += lambda / m .* Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
