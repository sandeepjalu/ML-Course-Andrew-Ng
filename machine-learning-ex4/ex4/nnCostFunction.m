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



%PART 1

% hx = X * theta;

% % sigmoid = arrayfun(@(t) 1/(1+1^(-t)), hx);
% h = sigmoid(hx);

% tempo = theta;
% tempo(1) = 0;
% temp = (h - y);
% grad = (1/m)*(X'*temp) + (lambda/m)*tempo;

% t_theta = theta(2:end,:);

% J = (1/m)*(-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*sum(t_theta .^2);

% [J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
% size(X)  % 5000 x 400
% size(Theta1) % 25 x 401
% size(Theta2) % 10 x 26
% size(y)

K = num_labels; % 10
tJ = zeros(K,1); % 10 x 1
X = [ones(m, 1) X];


a1 = sigmoid(X * Theta1');
a1 = [ones(m, 1) a1];

a2 = sigmoid(a1 * Theta2');
ny = zeros(m,K); %5000 x 10
for c = 1:m
	ny(c,y(c)) = 1;
end;

t1 = Theta1'(2:end,:); % 400 x 25
t2 = Theta2'(2:end,:); % 25 x 10

for c = 1:K
	tJ(c) = ((1/m)*(-ny(:,c)'*log(a2(:,c)) - (1-ny(:,c))'*log(1-a2(:,c))));
end;

extra = (lambda/(2*m))*(sum(sum(t1 .^2)) + sum(sum(t2 .^2)));

% nJ = ((1/m)*(-ny'*log(a2) - (1-ny)'*log(1-a2)));

% for c = 1:K
	% hx = X * theta;
	% sigmoid = arrayfun(@(t) 1/(1+1^(-t)), hx);
	% size(log(a2))
	% size(ny')
	% size(ssa)
	%a2 = [ones(m, 1) a2];
	% a2size = size(a2,2); % 10
	% size(y') %
	% for t = 1:a2size
	% 	% size(a2)  % 5000 x 10
	% 	% a2size   % 10
	% 	att = a2(:,t);
	% 	% size(att) % 5000 x 1
	% 	% size(y') % 1 x 5000
	% 	tJ(c) = tJ(c) + (1/m)*(-y'*log(att) - (1-y)'*log(1-att));
	% end;
% end;


J = sum(tJ) + extra










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
