import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt


def solve_quadratic(a, b, c):
    """
    Function to solve quadratic equations ax^2 + bx + c = 0
    :param a:
    :param b:
    :param c:
    :return: roots
    """
    a = complex(a)
    discriminant = b ** 2 - 4 * a * c
    return [(-b + np.sqrt(discriminant)) / (2 * a), (-b - np.sqrt(discriminant)) / (2 * a)]


class ERPCA(object):

    def __init__(self, observation_matrix: np.array,
                 eta_alpha: float = 0.5, eta_beta: float = 0.08,
                 rank_prior: float = 18, sparse_prior: float = 0.1,
                 selection_size: int = 20, runs: int = 100, train_size: int = 10):
        """
        Initialize the penalty parameters
        """

        self.rank_prior = rank_prior
        self.sparse_prior = sparse_prior
        self.selection_size = selection_size
        self.runs = runs

        if len(observation_matrix.shape) == 3:
            self.group_num = 1
            self.sample = np.expand_dims(observation_matrix, axis=-1)
        else:
            self.group_num = observation_matrix.shape[-1]
            self.sample = observation_matrix

        self.eta_alpha = eta_alpha
        self.eta_beta = eta_beta

        self.alpha = [None for _ in range(self.group_num)]
        self.beta = [None for _ in range(self.group_num)]
        self.mu = [None for _ in range(self.group_num)]

        self.L_matrix = None
        self.S_matrix_group = None

        for g in range(self.group_num):
            self.alpha[g], self.beta[g], self.mu[g] = \
                self.__tuning_parameters(self.sample[:, :, train_size, g], g)


        self.alpha_total = self.alpha[0]
        self.beta_total = self.beta[0]
        self.mu_total = self.mu[0]

    def __tuning_parameters(self, train_obs, group_i=None):
        """
        Hyperparameters Tuning for eRPCA
        :param train_obs: training observations for tunning parameters
        :param group_i: group indicator
        :return: tuning parameters
        """
        m_1 = train_obs.shape[0]
        m_2 = train_obs.shape[1]

        # Initialize parameters
        alpha_0 = 1
        beta_0 = 1 / np.sqrt(np.max((m_1, m_2)))
        mu_0 = (m_1 * m_2) / (4 * np.nansum(np.abs(train_obs)))

        # adjustment rate
        pass

        # arrays to store selections and results
        alpha_select = -100 * np.ones(self.selection_size)
        beta_select = -100 * np.ones(self.selection_size)
        rank_all = -100 * np.ones(self.selection_size)
        sparse_all = -100 * np.ones(self.selection_size)

        # Prior values for rank and sparsity
        pass

        # set initial values
        alpha_select[0] = alpha_0
        beta_select[0] = beta_0

        # Hyperparameter selection loop
        for i in range(0, self.selection_size - 1):
            if group_i is None:
                L_all, S_all, P_all, Y_all = self.__mc_function_mle(obs_test=self.sample,
                                                                    mu=mu_0,
                                                                    alpha=alpha_select[i],
                                                                    beta=beta_select[i])
            else:
                L_all, S_all, P_all, Y_all = self.__mc_function_mle(obs_test=self.sample[:, :, :, group_i],
                                                                    mu=mu_0,
                                                                    alpha=alpha_select[i],
                                                                    beta=beta_select[i])

            S_all[:, :, self.runs - 1][np.abs(S_all[:, :, self.runs - 1]) <= 1e-5] = 0

            rank_all[i] = np.linalg.matrix_rank(L_all[:, :, self.runs - 1])
            sparse_all[i] = np.sum(S_all[:, :, self.runs - 1] != 0) / (m_1 * m_2)

            # Adjust alpha and beta based on rank and sparsity
            if (rank_all[i] > self.rank_prior) & (sparse_all[i] > self.sparse_prior):
                alpha_select[i + 1] = alpha_select[i] + self.eta_alpha
                beta_select[i + 1] = beta_select[i] + self.eta_beta

            elif (rank_all[i] <= self.rank_prior) & (sparse_all[i] > self.sparse_prior):
                alpha_select[i + 1] = alpha_select[i]
                beta_select[i + 1] = beta_select[i] + self.eta_beta

            elif (rank_all[i] > self.rank_prior) & (sparse_all[i] <= self.sparse_prior):
                alpha_select[i + 1] = alpha_select[i] + self.eta_alpha
                beta_select[i + 1] = beta_select[i]

            elif (rank_all[i] <= self.rank_prior) & (sparse_all[i] <= self.sparse_prior):
                alpha_select[i + 1] = alpha_select[i]
                beta_select[i + 1] = beta_select[i]

            # Break loop if no change in hyperparameters
            if (alpha_select[i + 1] == alpha_select[i]) & (beta_select[i + 1] == beta_select[i]):
                break

        m_1 = train_obs.shape[0]
        m_2 = train_obs.shape[1]

        return (alpha_select[alpha_select != -100][-1],
                beta_select[beta_select != -100][-1],
                (m_1 * m_2) / (4 * np.nansum(np.abs(train_obs))))

    def __mc_function_mle(self, obs_test: np.array, mu: np.array,
                          alpha: np.array, beta: np.array, L=None):
        """
        Main function for RPCA
        :param obs_test: observations
        :param mu:
        :param alpha:
        :param beta:
        :param L:
        :return:
        """
        m_1 = obs_test.shape[0]
        m_2 = obs_test.shape[1]

        # initialization

        # create empty arrays to store the results
        P_all = np.zeros((m_1, m_2, self.runs))
        S_all = np.zeros((m_1, m_2, self.runs))
        L_all = np.zeros((m_1, m_2, self.runs))
        Y_all = np.zeros((m_1, m_2, self.runs))

        is_L_given = False

        if L is None:
            obs_test_mean = np.nanmean(obs_test, axis=2)
            U, S, Vh = svd(obs_test_mean, full_matrices=False)
            L = U @ np.diag(
                np.append(S[0: 2], np.zeros((min((m_1, m_2)) - 2)))
            ) @ Vh
            L_all[:, :, 0] = L

        else:
            L_all[:, :, :] = L
            is_L_given = True

        Y = np.zeros((m_1, m_2))
        P = np.ones((m_1, m_2))

        # initialize the first iteration
        P_all[:, :, 0] = P
        Y_all[:, :, 0] = Y

        return self.__mc_samp(P_all, S_all, Y_all, L_all, obs_test, alpha, beta, mu, is_L_given)

    def __mc_samp(self, P_all, S_all, Y_all, L_all, obs_test, alpha, beta, mu, is_L_given):
        """
        Monte Carlo sampling for ERPCA
        :param P_all:
        :param S_all:
        :param Y_all:
        :param L_all:
        :param obs_test: observations
        :param alpha:
        :param beta:
        :param mu:
        :param is_L_given: whether update L
        :return: result
        """
        # Update S matrix
        for i in range(0, self.runs - 1):
            S_new = self.__update_S(P_all[:, :, i], L_all[:, :, i], Y_all[:, :, i], beta, mu)
            S_all[:, :, i + 1] = S_new

            # Update L matrix
            if is_L_given:
                pass
            else:
                L_new = self.__update_L(P_all[:, :, i], S_all[:, :, i + 1], Y_all[:, :, i], alpha, mu)
                L_all[:, :, i + 1] = L_new

            # Update P matrix
            P_new = self.__update_P(L_all[:, :, i + 1], S_all[:, :, i + 1], Y_all[:, :, i], obs_test,
                                    m1=obs_test.shape[0], m2=obs_test.shape[1], mu=mu)
            P_all[:, :, i + 1] = P_new

            # Update Y matrix
            Y_new = Y_all[:, :, i] + mu * (P_all[:, :, i + 1] - L_all[:, :, i + 1] - S_all[:, :, i + 1])
            Y_all[:, :, i + 1] = Y_new

        return (L_all, S_all, P_all, Y_all)

    def __update_S(self, P_k, L_k, Y_k, beta, mu):
        """
        Function to update matrix S
        :param P_k:
        :param L_k:
        :param Y_k:
        :param beta:
        :param mu:
        :return: new matrix S
        """
        X = P_k - L_k + (1 / mu) * Y_k
        tau = beta / mu

        # Apply soft thresholding
        S_new = np.sign(X) * np.maximum(np.abs(X) - tau, 0)
        return S_new

    def __update_L(self, P_k, S_k, Y_k, alpha, mu):
        """
        Function to update matrix L
        :param P_k: 
        :param S_k: 
        :param Y_k: 
        :param alpha: 
        :param mu:
        :return: new matrix L
        """
        X = P_k - S_k + (1 / mu) * Y_k

        # Perform SVD on X
        U, S, Vh = svd(X, full_matrices=False)

        tau = alpha / mu
        d = np.diag(S)

        S_d = np.sign(d) * np.maximum(np.abs(d) - tau, 0)

        L_new = U @ S_d @ Vh

        return L_new

    def __update_P(self, L_k, S_k, Y_k, obs_matrix, m1, m2, mu):
        """
        Function to update matrix P
        :param L_k:
        :param S_k:
        :param Y_k:
        :param obs_matrix: observations
        :param m1:
        :param m2:
        :param mu:
        :return: new matrix P
        """
        P_new = np.zeros((m1, m2))
        for i in range(0, m1):
            for j in range(0, m2):
                P_seq = np.arange(0.00001, 1.00001, 0.05)
                arg_func = (-np.mean(obs_matrix[i, j, :]) * np.log(P_seq) -
                            (1 - np.mean(obs_matrix[i, j, :])) * np.log(1 - P_seq) +
                            (mu / 2) * (P_seq - L_k[i, j] - S_k[i, j] + (1 / mu) * Y_k[i, j]) ** 2)
                P_new[i, j] = P_seq[np.argmin(arg_func)]

        return P_new

    def run(self):
        """
        Main function for recover matrix
        :return: recovered matrix
        """
        if self.group_num == 1:
            L_all, S_all, P_all, Y_all = self.__mc_function_mle(self.sample[:, :, :, 0],
                                                                mu=self.mu[0],
                                                                alpha=self.alpha[0],
                                                                beta=self.beta[0])
            S_all[:, :, self.runs - 1][np.abs(S_all[:, :, self.runs - 1]) <= 1e-5] = 0

            self.L_matrix = L_all[:, :, self.runs - 1]
            self.S_matrix_group = S_all[:, :, self.runs - 1]

            return self.L_matrix, self.S_matrix_group
        else:
            # through algorithm 1 to compute L_tilde
            d1, d2, d3, d4 = self.sample.shape
            L_all, _, _, _ = self.__mc_function_mle(self.sample.reshape(d1, d2, d3 * d4),
                                                    mu=self.mu[0],
                                                    alpha=self.alpha[0],
                                                    beta=self.beta[0])

            L_tilde = L_all[:, :, self.runs - 1]

            self.S_matrix_group = np.zeros((d1, d2, self.group_num))

            for g in range(self.group_num):
                _, S_all, _, _ = self.__mc_function_mle(self.sample[:, :, :, g],
                                                        mu=self.mu[g],
                                                        alpha=self.alpha[g],
                                                        beta=self.beta[g],
                                                        L=L_tilde)
                S_all[:, :, self.runs - 1][np.abs(S_all[:, :, self.runs - 1]) <= 1e-5] = 0

                self.S_matrix_group[:, :, g] = S_all[:, :, self.runs - 1]

            self.L_matrix = L_tilde
            return self.L_matrix, self.S_matrix_group


# Function to plot grayscale image
def plot_matrix(matrix, title):
    plt.imshow(matrix, cmap='gray', aspect='auto')
    plt.title(title, fontsize=16)
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    pass
