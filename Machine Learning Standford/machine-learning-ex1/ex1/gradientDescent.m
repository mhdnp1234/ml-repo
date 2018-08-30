function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta = theta;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % ============================================================

    % Save the cost J in every iteration    
    difference = (X * theta - y);
    %          97x2    2*1   97x1
    %theta0 = theta(1) - alpha * sum(difference.* X(:,1))/m;
    %theta1 = theta(2) - alpha * sum(difference.* X(:,2))/m;
    % theta = [theta0; theta1];
    % OPTIMIZED FORMULA FOR FINDING NEW THETA
    theta = theta - (X' * difference).*(alpha/m);
    J_history(iter) = computeCost(X, y, theta); 
end
end
