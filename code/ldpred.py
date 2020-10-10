import numpy as np
import scipy.stats
import sys
from time import time


def update_step_naive(beta_hat,
                      ld_mat,
                      vi_mu,
                      vi_s,
                      vi_psi,
                      sigma_sq_e,
                      sigma_sq_0,
                      sigma_sq_1,
                      p_0):
    new_mu = np.copy(vi_mu)
    new_s = np.copy(vi_s)
    new_psi = np.copy(vi_psi)
    for i in range(vi_mu.shape[0]):
        new_mu[i] = (beta_hat[i]
                     - ld_mat[i, :].dot(new_mu)
                     + new_mu[i] * ld_mat[i, i])
        new_mu[i] /= (new_psi[i] * sigma_sq_e / sigma_sq_0
                      + (1 - new_psi[i]) * sigma_sq_e / sigma_sq_1
                      + ld_mat[i, i])
        new_s[i] = 1 / (new_psi[i] / sigma_sq_0
                        + (1 - new_psi[i]) / sigma_sq_1
                        + ld_mat[i, i] / sigma_sq_e)

        raw_psi_0 = p_0 * np.exp(
            -0.5 * np.log(sigma_sq_0)
            - 0.5 / sigma_sq_0 * (new_mu[i] ** 2 + new_s[i])
        )
        raw_psi_1 = (1 - p_0) * np.exp(
            -0.5 * np.log(sigma_sq_1)
            - 0.5 / sigma_sq_1 * (new_mu[i] ** 2 + new_s[i])
        )
        new_psi[i] = raw_psi_0 / (raw_psi_0 + raw_psi_1)

    return new_mu, new_s, new_psi


def update_step_sparse(beta_hat,
                       ld_mat,
                       vi_mu,
                       vi_s,
                       vi_psi,
                       sigma_sq_e,
                       sigma_sq_1,
                       p_0):
    new_mu = np.copy(vi_mu)
    new_s = np.copy(vi_s)
    new_psi = np.copy(vi_psi)
    for i in range(vi_mu.shape[0]):
        this_mu = (beta_hat[i]
                   - ld_mat[i, :].dot(new_mu * (1 - new_psi))
                   + new_mu[i] * ld_mat[i, i] * (1 - new_psi[i]))
        this_mu /= sigma_sq_e / sigma_sq_1 + ld_mat[i, i]
        new_mu[i] = this_mu
        new_s[i] = 1 / (1 / sigma_sq_1 + ld_mat[i, i] / sigma_sq_e)

        psi_num = (p_0 / (1 - p_0)
                   * np.sqrt(1 + ld_mat[i, i] * sigma_sq_1 / sigma_sq_e)
                   * np.exp(-0.5 * (beta_hat[i]
                                    - ld_mat[i, :].dot(new_mu * (1 - new_psi))
                                    + new_mu[i] * ld_mat[i, i]
                                    * (1 - new_psi[i])) ** 2
                            / (sigma_sq_e ** 2 / sigma_sq_1
                               + sigma_sq_e * ld_mat[i, i])))
        new_psi[i] = psi_num / (1 + psi_num)
    return new_mu, new_s, new_psi


sigma_sq_1 = 1.0
sigma_sq_e = float(sys.argv[1])
num_reps = int(sys.argv[2])

p_zero = 0.99
num_sites = 1000

mse_mat = np.zeros((num_reps, 10))
cor_mat = np.zeros((num_reps, 10))
header = ['beta_hat', 'MLE', 'naive_1.0', 'naive_1e-1', 'naive_1e-2',
          'naive_1e-3', 'naive_1e-4', 'naive_1e-5', 'naive_1e-10', 'sparse']

for rep in range(num_reps):
    print(rep)
    true_beta = np.zeros(num_sites)
    nonzero = np.random.choice([True, False], num_sites, p=[1-p_zero, p_zero])
    true_beta[nonzero] = np.random.normal(loc=0,
                                          scale=np.sqrt(sigma_sq_1),
                                          size=nonzero.sum())

    ld_matrix = (scipy.stats.wishart.rvs(num_sites, np.eye(num_sites))
                 / num_sites)

    chol = np.linalg.cholesky(ld_matrix)
    inv = np.linalg.inv(ld_matrix)
    noise = chol.dot(np.random.normal(loc=0,
                                      scale=np.sqrt(sigma_sq_e),
                                      size=num_sites))
    beta_hat = ld_matrix.dot(true_beta) + noise

    cor_mat[rep, 0] = np.corrcoef(beta_hat, true_beta)[0, 1]
    cor_mat[rep, 1] = np.corrcoef(inv.dot(beta_hat), true_beta)[0, 1]
    mse_mat[rep, 0] = np.mean((beta_hat - true_beta)**2)
    mse_mat[rep, 1] = np.mean((inv.dot(beta_hat) - true_beta)**2)

    for idx, sigma_0 in enumerate([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-10]):
        start_time = time()
        vi_mu = np.zeros_like(beta_hat)
        vi_s = (sigma_sq_1 + sigma_sq_e) * np.ones(num_sites)
        vi_psi = np.ones(num_sites)
        for i in range(100):
            vi_mu, vi_s, vi_psi = update_step_naive(beta_hat,
                                                    ld_matrix,
                                                    vi_mu,
                                                    vi_s,
                                                    vi_psi,
                                                    sigma_sq_e,
                                                    sigma_0,
                                                    sigma_sq_1,
                                                    p_zero)
        cor_mat[rep, idx + 2] = np.corrcoef(vi_mu, true_beta)[0, 1]
        mse_mat[rep, idx + 2] = np.mean((vi_mu - true_beta)**2)
        print('\tScheme took', time() - start_time)
    vi_mu = np.zeros_like(beta_hat)
    vi_s = (sigma_sq_1 + sigma_sq_e) * np.ones(num_sites)
    vi_psi = p_zero * np.ones(num_sites)
    start_time = time()
    for i in range(100):
        vi_mu, vi_s, vi_psi = update_step_sparse(beta_hat,
                                                 ld_matrix,
                                                 vi_mu,
                                                 vi_s,
                                                 vi_psi,
                                                 sigma_sq_e,
                                                 sigma_sq_1,
                                                 p_zero)
    print('\tScheme took', time() - start_time)
    cor_mat[rep, -1] = np.corrcoef(vi_mu * (1 - vi_psi), true_beta)[0, 1]
    mse_mat[rep, -1] = np.mean((vi_mu * (1 - vi_psi) - true_beta)**2)

np.savetxt('../data/ldpred/cor_mat_' + str(sigma_sq_e) + '.txt', cor_mat,
           header='\t'.join(header))
np.savetxt('../data/ldpred/mse_mat_' + str(sigma_sq_e) + '.txt', mse_mat,
           header='\t'.join(header))
