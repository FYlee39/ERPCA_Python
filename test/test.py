from eRPCA_py import eRPCA
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt


def generate_L_S(p: int, mu: float, sigma: float,
                 lower: float, upper: float, group: int = 1):
    """
    Generate the low-rank matrix L and the sparse matrix S
    :param p: number of dimensions
    :param mu: mean of the gaussian distribution
    :param sigma: standard deviation of the gaussian distribution
    :param lower: lower bound
    :param upper: upper bound
    :param group: number of groups
    :return: L, S
    """
    L = np.random.normal(mu, sigma, (p, p))
    U, S, Vh = svd(L, full_matrices=False)
    S[int(p / 5):] = 0
    L = U @ np.diag(S) @ Vh
    S = [None] * group
    for g in range(group):
        indices = np.random.choice(p * p, int(p * p / 20), replace=False)
        S[g] = np.zeros(p * p)
        S[g][indices] = 1
        S[g] = S * np.random.uniform(low=lower, high=upper, size=p * p)
        S[g] = S[g].reshape((p, p))
    return L, S


def single_group(p: int = 10, mu: float = 0, sigma: float = 1,
                 lower: float = 0, upper: float = 1,
                 n: int = 500, rep_num: int = 30, type: str = "Bernoulli"):
    """
    Perform a numerical experiments for certain dimmension
    :param p: number of dimensions
    :param mu: mean of the gaussian distribution
    :param sigma: standard deviation of the gaussian distribution
    :param lower: lower bound
    :param upper: upper bound
    :param n: number of samples
    :param rep_num: number of replications
    :return: the errors of L and S
    """
    L_error = np.zeros(rep_num)
    S_error = np.zeros(rep_num)
    for rep in range(rep_num):
        L, S = generate_L_S(p, mu=mu, sigma=sigma, lower=lower, upper=upper)
        if len(S) == 1:
            # single group
            S = S[0]
        theta = L + S
        M = np.zeros((p, p, n))
        for i in range(0, theta.shape[0]):
            for j in range(0, theta.shape[1]):
                if type == "Bernoulli":
                    M[i, j, :] = np.random.binomial(n=1, p=theta[i, j], size=n)
                elif type == "Exponential":
                    M[i, j, :] = np.random.exponential(size=n, scale=1 / theta[i, j])
                elif type == "Poisson":
                    M[i, j, :] = np.random.poisson(lam=theta[i, j], size=n)
                else:
                    pass

        erpca = eRPCA.ERPCA(observation_matrix=M)
        L_est, S_est = erpca.run()
        L_error[rep] = np.linalg.norm(L - L_est, ord="fro")
        S_error[rep] = np.linalg.norm(S - S_est, ord="fro")

    return L_error, S_error



if __name__ == "__main__":

    # single-group setting
    p_list = [10, 20, 30]
    type_list = ["Bernoulli", "Exponential", "Poisson"]

    L_1, S_1 = single_group(p=p_list[0], type=type_list[0], mu=0.5, sigma=0.15, lower=0.2, upper=0.3)
    L_2, S_2 = single_group(p=p_list[1], type=type_list[1], mu=0.5, sigma=0.15, lower=0.2, upper=0.3)
    L_3, S_3 = single_group(p=p_list[2], type=type_list[2], mu=0.5, sigma=0.15, lower=0.2, upper=0.3)

    # multi-group
    pass