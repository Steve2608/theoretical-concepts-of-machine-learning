import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

data = np.loadtxt("cnvdata.csv", delimiter=",", skiprows=1)

plt.figure(figsize=(16, 9))
plt.hist(data, bins=50)
plt.title('Data distribution')
plt.ylabel('$p(X)$')
plt.xlabel('$X$')
plt.show()


class EM:
    def __init__(self, X, K, max_iter, verbose=False):
        self.X = X
        self.K = K
        self.max_iter = max_iter
        self.verbose = verbose

    def poisson(self, l, x_i):
        return stats.poisson(l).pmf(x_i)

    def alpha_optimize(self, R_ik, k) -> np.real:
        """Optimizing for one alpha in k-th column"""
        return R_ik[:, k].sum() / len(R_ik)

    def alphas(self, R_ik) -> [np.real]:
        """Optimizing over all alphas"""
        return np.asarray([self.alpha_optimize(R_ik, k) for k in range(R_ik.shape[1])])

    def lambda_optimize(self, R_ik, X, k) -> np.real:
        """Optimizing for one lambda_k:
        lambda_k = sum(R_ik * x_i) / sum(R_ik)"""
        return (R_ik[:, k] * X).sum() / R_ik[:, k].sum()

    def lambdas(self, R_ik, X) -> [np.real]:
        """Optimizing over all lambdas"""
        return np.asarray([self.lambda_optimize(R_ik, X, k) for k in range(R_ik.shape[1])])

    def r_ik_optimize(self, X, i, K, k, L, A) -> [np.real, np.real]:
        """Optimizing for one r_ik"""
        return A[k] * self.poisson(L[k], X[i]) / np.asarray(
            [A[l] * self.poisson(lam, X[i]) for l, lam in zip(range(K), L)]).sum()

    def M_step(self, R_ik, X):
        return self.alphas(R_ik), self.lambdas(R_ik, X)

    def E_step_initial(self):
        return np.random.rand(self.X.shape[0], self.K)

    def E_step(self, X, K, L, A):
        R_ik = np.zeros((X.shape[0], K))
        # Iterating over all x_i
        for i in range(X.shape[0]):
            # iterating over all ks
            for k in range(K):
                R_ik[i, k] = self.r_ik_optimize(X, i, K, k, L, A)
        return R_ik

    def does_converge(self, L_curr, L_prev) -> bool:
        return np.allclose(L_curr, L_prev, rtol=1e-06, atol=1e-06)

    def EM(self):
        L_curr = np.zeros(self.K)
        R_ik = self.E_step_initial()

        for i in range(self.max_iter):
            if self.verbose:
                print(f'Iteration {i + 1}/{self.max_iter}')
            L_prev = L_curr.copy()
            A, L_curr = self.M_step(R_ik, self.X)
            R_ik = self.E_step(self.X, self.K, L_curr, A)
            if self.does_converge(L_curr, L_prev):
                if self.verbose:
                    print(f'EM reached convergence in step {i + 1}!')
                return A, L_curr

        return L_curr, A

    def __str__(self):
        result = self.EM()
        result_as_list = [(l, a) for l, a in zip(result[0], result[1])]
        result_sorted = np.asarray(sorted(result_as_list))
        return f'Lambdas: {result_sorted[:, 0]}\nAlphas: {result_sorted[:, 1]}'
