import numpy as np


classical = []
oracle = []
new_vi = []
naive05 = []
naive01 = []
naive005 = []


for rep in range(1, 6):
    print(rep)
    truth = np.load('../data/sparse_pca/data_denoised_rep{}.npy'.format(rep))
    data = np.load('../data/sparse_pca/data_rep{}.npy'.format(rep))
    keep = np.load('../data/sparse_pca/keep_rep{}.npy'.format(rep))

    # classical
    u, s, v = np.linalg.svd(data)
    reconst = (u[:, 0:2] * s[:2]).dot(v[:2, :])
    classical.append(np.sum((reconst - truth)**2))

    # oracle
    u, s, v = np.linalg.svd(data[:, keep])
    reconst = np.zeros_like(data)
    reconst[:, keep] = (u[:, 0:2] * s[:2]).dot(v[:2, :])
    oracle.append(np.sum((reconst-truth)**2))

    print('\t Done with SVDs')
    # naive schemes
    w = np.load('../data/sparse_pca/vi_mu_w_0.05_rep{}.npy'.format(rep))
    z = np.load('../data/sparse_pca/vi_mu_z_0.05_rep{}.npy'.format(rep))
    reconst = z.dot(w.T)
    naive05.append(np.sum((reconst-truth)**2))

    w = np.load('../data/sparse_pca/vi_mu_w_0.01_rep{}.npy'.format(rep))
    z = np.load('../data/sparse_pca/vi_mu_z_0.01_rep{}.npy'.format(rep))
    reconst = z.dot(w.T)
    naive01.append(np.sum((reconst-truth)**2))

    w = np.load('../data/sparse_pca/vi_mu_w_0.005_rep{}.npy'.format(rep))
    z = np.load('../data/sparse_pca/vi_mu_z_0.005_rep{}.npy'.format(rep))
    reconst = z.dot(w.T)
    naive005.append(np.sum((reconst-truth)**2))

    # new vi_scheme
    w = np.load('../data/sparse_pca/vi_mu_w_sparse_rep{}.npy'.format(rep))
    z = np.load('../data/sparse_pca/vi_mu_z_sparse_rep{}.npy'.format(rep))
    psi = np.load('../data/sparse_pca/vi_psi_0_sparse_rep{}.npy'.format(rep))
    reconst = z.dot((w * (1-psi)).T)
    new_vi.append(np.sum((reconst-truth)**2))

print('Classical', np.min(classical), np.mean(classical), np.max(classical))
print('Oracle', np.min(oracle), np.mean(oracle), np.max(oracle))
print('Naive 0.05', np.min(naive05), np.mean(naive05), np.max(naive05))
print('Naive 0.01', np.min(naive01), np.mean(naive01), np.max(naive01))
print('Naive 0.005', np.min(naive005), np.mean(naive005), np.max(naive005))
print('New VI', np.min(new_vi), np.mean(new_vi), np.max(new_vi))
