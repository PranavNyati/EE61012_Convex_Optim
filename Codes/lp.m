%% LP Ex1

x = sdpvar(3, 1);
Constraints  = [x >= 0, 2*x(1) + 3*x(2) + x(3) == 1];
Objective = x(1) - x(2);
ops = sdpsettings('solver', 'linprog', 'verbose', 1, 'debug', 1);
optimize(Constraints, Objective, ops)

plot_x = [0 1/2 0];
plot_y = [0 0 1/3];
 %%fill(plot_x, plot_y, 'r')

 %% Dual of LP
 A = [2, 3, 1];
 c = [1 -1 0]';
 b = 1;

 y = sdpvar(1, 1);
 %% Constraints = (y <= -1/3);
 Constraints = transpose(A)*y <= c;
 Objective = -transpose(b)*y;
 ops = sdpsettings('solver', 'linprog', 'verbose', 1, 'debug', 1);
 optimize(Constraints, Objective, ops)
 
 value(x)
 
 value(y)

