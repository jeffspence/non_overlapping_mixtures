import numpy as np
import glob


cors = {}
mses = {}

its = set([])
sigmas = set([])

for fname in glob.glob('../data/ldpred/nimble_results_*'):
    mse = None
    cor = None
    read_mse = False
    read_cor = False
    with open(fname) as fh:
        for line in fh:
            if read_mse:
                mse = float(line.split()[-1])
                read_mse = False
            if read_cor:
                cor = float(line.split()[-1])
                read_cor = False
            if 'CORRELATION' in line:
                read_cor = True
            if 'MSE' in line:
                read_mse = True
    iteration = fname.split('_results_')[1].split('_')[0]
    sigma_sq = fname.split('_results_')[1].split('_')[1][:-4]
    its.add(int(iteration))
    sigmas.add(sigma_sq)
    cors[(iteration, sigma_sq)] = cor
    mses[(iteration, sigma_sq)] = mse


to_save_mse = np.zeros((len(its), len(sigmas)))
to_save_cor = np.zeros((len(its), len(sigmas)))
for i, idx in enumerate(sorted(its)):
    for j, s in enumerate(sorted(sigmas)):
        to_save_mse[i, j] = mses[(str(idx), s)]
        to_save_cor[i, j] = cors[(str(idx), s)]

np.save('../data/ldpred/nimble_mse_table.npy', to_save_mse)
np.save('../data/ldpred/nimble_cor_table.npy', to_save_cor)
