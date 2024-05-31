close all; clear; clc;


D=load("Kernel_SVM_Data.mat");
X=D.X;

X=transpose(X);
train_X=X(1:120, :);
test_X=X(121:150, :);

Y=D.Y;
train_Y=Y(1:120);
test_Y=Y(121:150, :);

% size(train_X);
% train_X;
% size(train_Y);
% train_Y;

% SVM DUAL

N=120;
c=10;
alpha=10;

lambda=sdpvar(120, 1);

Constraints = [lambda>=0, lambda<=alpha, transpose(lambda)*train_Y==0];
objective=0.0;

A=zeros([N,N]);
for i=1:N
    objective=objective-(lambda(i));
    for j=1:N
        objective=objective+((0.5)*lambda(i)*lambda(j)*train_Y(i)*train_Y(j)*(kernel(train_X(i, :), train_X(j, :), c)));
    end
end

optimize(Constraints, objective);
lambda_opt=value(lambda);
lambda_opt;



b_opt=0;
for i=1:N
    if (lambda_opt(i) >= 0.001) && (lambda_opt(i) < 10-0.001)
        b_opt=(1/train_Y(i));
        for j=1:N
            b_opt=b_opt-(lambda_opt(j)*train_Y(j)*(kernel(train_X(i, :), train_X(j, :), c)));
        end
%         - dot(train_Y.*lambda_opt, A(i, :));

%         i;
%         lambda_opt(i);
        b_opt
        break;
    end
end

train_error_cnt=0;
test_error_cnt=0;


% train error
for i=1:N
    y_hat=b_opt;
    for j=1:N
        y_hat = y_hat + (lambda_opt(j)*train_Y(j)*(kernel(train_X(i, :), train_X(j, :), c)));
    end
    if y_hat*train_Y(i) < 0
        train_error_cnt=train_error_cnt+1;
    end
end

train_error_cnt


% test error
for i=1:30
    y_hat=b_opt;
    for j=1:N
        y_hat = y_hat + (lambda_opt(j)*train_Y(j)*(kernel(test_X(i, :), train_X(j, :), c)));
    end
    if y_hat*test_Y(i) < 0
        test_error_cnt=test_error_cnt+1;
    end
end

test_error_cnt


function k=kernel(x_i, x_j, c)
%     k=exp(-c*(norm(train_X(i, :)-train_X(j, :)))^2);
    k=exp(-c*(norm(x_i-x_j))^2);
end 


