import numpy as np
import glob


cors = {}
mses = {}
its = set([])
sigmas = set([])
for fname in glob.glob('../data/ldpred/pyro_discrete*mse.txt'):
    iteration = fname.split('pyro_discrete_')[1].split('_')[2]
    sigma_sq = fname.split('pyro_discrete_')[1].split('_')[0]
    its.add(int(iteration))
    sigmas.add(sigma_sq)
    mses[(iteration, sigma_sq)] = np.loadtxt(fname)
for fname in glob.glob('../data/ldpred/pyro_discrete*cor.txt'):
    iteration = fname.split('pyro_discrete_')[1].split('_')[2]
    sigma_sq = fname.split('pyro_discrete_')[1].split('_')[0]
    its.add(int(iteration))
    sigmas.add(sigma_sq)
    cors[(iteration, sigma_sq)] = np.loadtxt(fname)

to_save_mse = np.zeros((len(its), len(sigmas)))
to_save_cor = np.zeros((len(its), len(sigmas)))
for i, idx in enumerate(sorted(its)):
    for j, s in enumerate(sorted(sigmas)):
        to_save_mse[i, j] = mses[(str(idx), s)]
        to_save_cor[i, j] = cors[(str(idx), s)]

np.save('../data/ldpred/pyro_mse_table.npy', to_save_mse)
np.save('../data/ldpred/pyro_cor_table.npy', to_save_cor)
