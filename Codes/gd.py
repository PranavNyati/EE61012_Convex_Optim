import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

c=40
NUM_ITERS = 100

def func(x):
    global c
    return x[0]**2 + c*x[1]**2

def grad(x):
    global c
    # return np.array([2*x(1), 2*c*x(2)])
    return np.array([2*x[0], 2*c*x[1]], dtype=float)

def gd(x_start, eta, n_iters):
    x = x_start.copy()
    x = x.astype(float)
    # x_hist = [x]
    x_hist=np.zeros((n_iters+1, 2))
    for _ in range(n_iters):
        print("Iteration: ", _)
        print("current x: ", x)
        x_hist[_, :] = x
        # if _ < 10:

        #     print(x_hist[:_+1, :])
        grad_ = grad(x)
        # print(grad_.shape, x.shape, eta)
        x -= eta * grad_
        # x -= eta * 
        # x_hist.append(x)
    x_hist[n_iters] = x
    return np.array(x_hist)

# x = np.array([2, NUM_ITERS])
beta = 2*c
eta = 1.0/beta
# eta = eta.astype(float)

x_start = np.array([100, 100])
x_hist = gd(x_start, eta, NUM_ITERS)
print(x_hist)

# # Plot the convergence on the contours of the function in 2D in the x-y plane
# x = np.linspace(0, 120, 10)
# y = np.linspace(0, 120, 10)
# X, Y = np.meshgrid(x, y)
# # Z = func(np.array([X, Y]))
# Z = X**2 + c*Y**2

# plt.contour(X, Y, Z, 20)
# x_hist = np.array(x_hist)
# plt.plot(x_hist.T[0], x_hist.T[1], 'b-')
# plt.plot(x_hist.T[0], x_hist.T[1], 'b.')
# plt.show()

# plot the convergence of the function as a function of the number of iterations
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
# z_hist = []
# for i in range(0, NUM_ITERS):
#     z = func(x_hist[i])
#     z_hist.append(z)

# z = np.array(z_hist)

z = func(x_hist.T)
print(z)
    
plt.plot(np.log(z), 'b-')
plt.show()
