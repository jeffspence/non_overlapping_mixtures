import numpy as np
import sys
from time import time


def update_step_naive(X,
                      vi_mu_z,
                      vi_sigma_z,
                      vi_psi_0,
                      vi_mu_w,
                      vi_sigma_w,
                      sigma_sq_e,
                      sigma_sq_0,
                      sigma_sq_1,
                      p_0):

    # Update Z
    new_mu_z = np.copy(vi_mu_z)
    expected_w_gram = np.einsum('ik,il->kl', vi_mu_w, vi_mu_w)
    expected_w_gram += np.diag(vi_sigma_w.sum(axis=0))
    expected_w = np.copy(vi_mu_w)
    new_sigma_z = np.linalg.inv(
        expected_w_gram / sigma_sq_e
        + np.eye(expected_w_gram.shape[0])
    )
    for n in range(X.shape[0]):
        new_mu_z[n] = new_sigma_z.dot(expected_w.T.dot(X[n])) / sigma_sq_e

    # Update W and Y
    new_mu_w = np.copy(vi_mu_w)
    new_sigma_w = np.copy(vi_sigma_w)
    new_psi = np.copy(vi_psi_0)
    expected_x_z_sum = np.einsum('ni,nk->ik', X, new_mu_z)
    expected_z_cov = np.einsum('nk,nl->kl', new_mu_z, new_mu_z)
    expected_z_cov += new_mu_z.shape[0] * new_sigma_z
    expected_z_sq_sum = np.diag(expected_z_cov)
    for i in range(vi_mu_w.shape[0]):
        for k in range(vi_mu_w.shape[1]):
            log_odds = (-0.5 * (1. / sigma_sq_0 + 1. / sigma_sq_1)
                        * (new_mu_w[i, k] ** 2 + new_sigma_w[i, k])
                        - 0.5 * np.log(sigma_sq_0 / sigma_sq_1)
                        + np.log(p_0 / (1-p_0)))
            new_psi[i, k] = 1. / (1 + np.exp(-log_odds))
            linked_ests = np.dot(new_mu_w[i], expected_z_cov[k])
            linked_ests -= new_mu_w[i, k] * expected_z_cov[k, k]
            new_sigma_w[i, k] = (expected_z_sq_sum[k] / sigma_sq_e
                                 + new_psi[i, k] / sigma_sq_0
                                 + (1 - new_psi[i, k]) / sigma_sq_1) ** -1
            new_mu_w[i, k] = (new_sigma_w[i, k]
                              * (expected_x_z_sum[i, k] - linked_ests)
                              / sigma_sq_e)

    return new_mu_z, new_sigma_z, new_psi, new_mu_w, new_sigma_w


def update_step_sparse(X,
                       vi_mu_z,
                       vi_sigma_z,
                       vi_psi_0,
                       vi_mu_w,
                       vi_sigma_w,
                       sigma_sq_e,
                       sigma_sq_1,
                       p_0):
    # Update Z
    new_mu_z = np.copy(vi_mu_z)
    expected_w = (1 - vi_psi_0) * vi_mu_w
    expected_w_gram = np.einsum('ik,il->kl',
                                expected_w,
                                expected_w)
    var_w = (1 - vi_psi_0) * (vi_mu_w ** 2 + vi_sigma_w)
    var_w -= expected_w ** 2
    expected_w_gram += np.diag(var_w.sum(axis=0))
    new_sigma_z = np.linalg.inv(expected_w_gram / sigma_sq_e
                                + np.eye(expected_w_gram.shape[0]))
    for n in range(X.shape[0]):
        new_mu_z[n] = new_sigma_z.dot(expected_w.T.dot(X[n])) / sigma_sq_e

    # Update W
    new_mu_w = np.copy(vi_mu_w)
    new_sigma_w = np.copy(vi_sigma_w)
    new_psi = np.copy(vi_psi_0)
    # expected_z_sq_sum = np.einsum('nk,kk->k', new_mu_z**2, new_sigma_z)
    expected_x_z_sum = np.einsum('ni,nk->ik', X, new_mu_z)
    expected_z_cov = np.einsum('nk,nl->kl', new_mu_z, new_mu_z)
    expected_z_cov += new_mu_z.shape[0] * new_sigma_z
    expected_z_sq_sum = np.diag(expected_z_cov)
    for i in range(vi_mu_w.shape[0]):
        for k in range(vi_mu_w.shape[1]):
            new_sigma_w[i, k] = (expected_z_sq_sum[k] / sigma_sq_e
                                 + 1. / sigma_sq_1) ** -1
            linked_ests = np.dot((1 - new_psi[i]) * new_mu_w[i],
                                 expected_z_cov[k])
            linked_ests -= ((1 - new_psi[i, k])
                            * new_mu_w[i, k]
                            * expected_z_cov[k, k])
            new_mu_w[i, k] = (new_sigma_w[i, k]
                              * (expected_x_z_sum[i, k] - linked_ests)
                              / sigma_sq_e)
            log_odds = (np.log(p_0 / (1-p_0))
                        + 0.5 * np.log(sigma_sq_1)
                        - 0.5 * new_mu_w[i, k]**2 / new_sigma_w[i, k]
                        - 0.5 * np.log(new_sigma_w[i, k]))
            new_psi[i, k] = 1. / (1 + np.exp(-log_odds))
    return new_mu_z, new_sigma_z, new_psi, new_mu_w, new_sigma_w


N = 500
num_vars = 10000


sigma_sq_e = 1.0
sigma_sq_1 = 0.5
num_non_zero = int(sys.argv[1])

if len(sys.argv) > 2:
    rep = '_rep' + sys.argv[2]
else:
    rep = ''


p_zero = 1. - num_non_zero / num_vars

clust = np.array([0] * 200 + [1] * 200 + [2] * 50 + [3] * 50)
data = np.random.normal(size=(N, num_vars))

keep = np.random.choice(num_vars, size=num_non_zero,
                        replace=False)
data_denoised = np.zeros_like(data)
for c in range(4):
    c_idx = np.ix_(clust == c, keep)
    data_denoised[c_idx] += np.random.normal(size=num_non_zero)

data = data + data_denoised
centerer = data.mean(axis=0, keepdims=True)
data -= centerer
normalizer = np.std(data, axis=0, keepdims=True)
data /= normalizer

data_denoised_centered = np.copy(data_denoised)
data_denoised_centered[:, keep] -= centerer[:, keep]
data_denoised_centered[:, keep] /= normalizer[:, keep]

np.save('../data/sparse_pca/keep' + rep + '.npy', keep)
np.save('../data/sparse_pca/data_denoised' + rep + '.npy',
        data_denoised_centered)
np.save('../data/sparse_pca/data' + rep + '.npy', data)

u, s, v = np.linalg.svd(data)
for sigma_sq_0 in [0.005, 0.01, 0.05]:
    print('Running naive with sigma_sq_0 = ', sigma_sq_0)
    start_time = time()
    vi_mu_z = np.copy(u[:, 0:2])
    vi_sigma_z = np.eye(2)
    vi_mu_w = np.copy(v[0:2, :].T) * s[0:2]
    vi_sigma_w = np.ones((num_vars, 2))
    vi_psi_0 = np.ones((num_vars, 2)) * 1e-10
    for i in range(250):
        (vi_mu_z,
         vi_sigma_z,
         vi_psi_0,
         vi_mu_w,
         vi_sigma_w) = update_step_naive(data,
                                         vi_mu_z,
                                         vi_sigma_z,
                                         vi_psi_0,
                                         vi_mu_w,
                                         vi_sigma_w,
                                         1.0,
                                         sigma_sq_0,
                                         sigma_sq_1,
                                         p_zero)

    print('Run took ', time() - start_time)
    np.save('../data/sparse_pca/vi_mu_z_' + str(sigma_sq_0) + rep + '.npy',
            vi_mu_z)
    np.save('../data/sparse_pca/vi_mu_w_' + str(sigma_sq_0) + rep + '.npy',
            vi_mu_w)

print('RUNNING SPARSE')
start_time = time()
vi_mu_z = np.copy(u[:, 0:2])
vi_sigma_z = np.eye(2)
vi_mu_w = np.copy(v[0:2, :].T) * s[0:2]
vi_sigma_w = np.ones((num_vars, 2))
vi_psi_0 = np.ones((num_vars, 2)) * 1e-10
for i in range(250):
    (vi_mu_z,
     vi_sigma_z,
     vi_psi_0,
     vi_mu_w,
     vi_sigma_w) = update_step_sparse(data,
                                      vi_mu_z,
                                      vi_sigma_z,
                                      vi_psi_0,
                                      vi_mu_w,
                                      vi_sigma_w,
                                      1.0,
                                      sigma_sq_1,
                                      p_zero)
print('Run took ', time() - start_time)

np.save('../data/sparse_pca/vi_mu_z_sparse' + rep + '.npy', vi_mu_z)
np.save('../data/sparse_pca/vi_mu_w_sparse' + rep + '.npy', vi_mu_w)
np.save('../data/sparse_pca/vi_sigma_w_sparse' + rep + '.npy', vi_sigma_w)
np.save('../data/sparse_pca/vi_psi_0_sparse' + rep + '.npy', vi_psi_0)
