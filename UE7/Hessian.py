import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid


def hessian(X, w, bias):
    w_bias = np.append(w, bias)
    X_bias = np.c_[X, np.ones(X.shape[0])]

    beta = calc_beta(X_bias, w_bias)
    return np.asarray([
        [hession_pos(X_bias, 0, 0, beta), hession_pos(X_bias, 0, 1, beta)],
        [hession_pos(X_bias, 1, 0, beta), hession_pos(X_bias, 1, 1, beta)]
    ])


def calc_beta(X, w):
    return np.sum([deriv_sigmoid(w.T @ x_i) for x_i in X], axis=0)


def hession_pos(X, j, i, beta):
    return (X[:, j].T * beta) @ X[:, i]


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def dir_newton(H, g):
    return - (np.linalg.inv(H) @ g)


def logistic_gradient(X, y, w):
    s = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        s += X[i, :] * (y[i] - sigmoid(w.T @ X[i, :]))
    return -s


def ce_loss(X, y, w):
    return - sum([
        y_i * np.log(sigmoid(w.T @ x_i)) +
        (1 - y_i) * np.log(1 - sigmoid(w.T @ x_i)) for x_i, y_i in zip(X, y)
    ])


def newton_regression(X, y, max_iter=1000, bias=0.0, lr=0.01):
    w_old = np.random.rand(X.shape[1])
    losses = []

    for _ in range(max_iter):
        H = hessian(X, w_old, bias)
        g = logistic_gradient(X, y, w_old)
        w_new = w_old + lr * dir_newton(H, g)
        losses.append(ce_loss(X, y, w_new))

        w_old = w_new

    return w_old, losses


def logistic_regression(X, y, max_iter=1000, lr=0.01):
    w_old = np.random.rand(X.shape[1])
    losses = []

    for _ in range(max_iter):
        w_new = w_old - lr * logistic_gradient(X, y, w_old)
        losses.append(ce_loss(X, y, w_new))

        w_old = w_new

    return w_old, losses


def plot_results(losses, method):
    plt.figure(figsize=(16, 9))

    plt.plot(losses)
    plt.title(f'Loss {method} Regression')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.show()


###########################

from sklearn.model_selection import train_test_split


def labels_to_0_1(y):
    return (y + 1) / 2


data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
data[:, -1] = labels_to_0_1(data[:, -1])

X_train, X_test, y_train, y_test = train_test_split(
    data[:, :-1], data[:, -1], test_size=0.2, shuffle=True)

#########

w_reg, losses_reg = logistic_regression(X_train, y_train)
plot_results(losses_reg, 'GD')

#########

for bias in range(0, 12, 3):
    w_newton, losses_newton = newton_regression(X_train, y_train, bias=bias)
    plot_results(losses_newton, 'Newton Method')
    print(f'Bias: {bias}')

#########

from time import perf_counter
from scipy.optimize import minimize


def perform_mini(X, y, method, max_iter=1000, lr=0.01):
    losses, w_result = [], None
    w_old = np.random.rand(X_train.shape[1])
    start = perf_counter()

    if method in ['BFGS', 'CG']:
        w_result = minimize(lambda w, X, y: ce_loss(X, y, w),
                            x0=w_old, args=(X, y), method=method,
                            options={'maxiter': max_iter},
                            callback=lambda w: losses.append(ce_loss(X, y, w))
                            ).x
    elif method == 'GD':
        for _ in range(max_iter):
            w_new = w_old - lr * logistic_gradient(X, y, w_old)
            losses.append(ce_loss(X, y, w_new))

            # terminate if convergence reached
            # if np.allclose(w_new, w_old, atol=1e-4):
            #     break
            w_old = w_new

        w_result = w_old
    else:
        raise ValueError(f'Invalid Method passed: {method}')

    return w_old, losses, perf_counter() - start


def predict(X, w, clf=lambda x: x >= 0.5):
    return [clf(w.T @ x_i) for x_i in X]


def accuracy(y_actual, y_pred):
    return np.sum(y_pred == y_actual) / len(y_actual)


##########

w_bfgs, losses_bfgs, time_bfgs = perform_mini(X_train, y_train, 'BFGS')
w_cg, losses_cg, time_cg = perform_mini(X_train, y_train, 'CG')
w_gd, losses_gd, time_gd = perform_mini(X_train, y_train, 'GD')

plot_results(losses_bfgs, 'BFGS')
print(f'Optimal w found: {w_bfgs}')
print(f'Accuracy: {accuracy(y_test, predict(X_test, w_bfgs))}')
print(f'Time elapsed: {time_bfgs}')

plot_results(losses_cg, 'CG')
print(f'Optimal w found: {w_cg}')
print(f'Accuracy: {accuracy(y_test, predict(X_test, w_cg))}')
print(f'Time elapsed: {time_cg}')

plot_results(losses_gd, 'GD')
print(f'Optimal w found: {w_gd}')
print(f'Accuracy: {accuracy(y_test, predict(X_test, w_gd))}')
print(f'Time elapsed: {time_gd}')
