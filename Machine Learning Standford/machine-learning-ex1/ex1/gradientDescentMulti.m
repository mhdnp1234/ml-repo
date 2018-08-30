function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
                % 47x3    3*1   47x1
    %theta0 = theta(1) - alpha * sum(difference.* X(:,1))/m;
    %theta1 = theta(2) - alpha * sum(difference.* X(:,2))/m;
    %theta2 = theta(3) - alpha * sum(difference.* X(:,3))/m;
    
    % OPTIMIZED FORMULA FOR FINDING NEW THETA
    theta = theta - (X' * (X * theta - y)).*(alpha/m);
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
