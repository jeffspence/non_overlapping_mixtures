import numpy as np
import matplotlib.pyplot as plt

for rep in ['1', '2', '3', '4', '5']:
    print('Plotting Rep', rep)
    w005 = np.load('../data/sparse_pca/vi_mu_w_0.005_rep' + rep + '.npy')
    z005 = np.load('../data/sparse_pca/vi_mu_z_0.005_rep' + rep + '.npy')
    w01 = np.load('../data/sparse_pca/vi_mu_w_0.01_rep' + rep + '.npy')
    z01 = np.load('../data/sparse_pca/vi_mu_z_0.01_rep' + rep + '.npy')
    w05 = np.load('../data/sparse_pca/vi_mu_w_0.05_rep' + rep + '.npy')
    z05 = np.load('../data/sparse_pca/vi_mu_z_0.05_rep' + rep + '.npy')
    keep = np.load('../data/sparse_pca/keep_rep' + rep + '.npy')
    data = np.load('../data/sparse_pca/data_rep' + rep + '.npy')
    w_sparse = np.load('../data/sparse_pca/vi_mu_w_sparse_rep' + rep + '.npy')
    w_sparse_sigma = np.load(
        '../data/sparse_pca/vi_sigma_w_sparse_rep' + rep + '.npy'
    )
    z_sparse = np.load('../data/sparse_pca/vi_mu_z_sparse_rep' + rep + '.npy')
    psi_sparse = np.load(
        '../data/sparse_pca/vi_psi_0_sparse_rep' + rep + '.npy'
    )

    z005 /= np.sqrt((z005**2).sum())
    z01 /= np.sqrt((z01**2).sum())
    z05 /= np.sqrt((z05**2).sum())
    z_sparse /= np.sqrt((z_sparse**2).sum())

    plt.scatter(z_sparse[0:200, 0], z_sparse[0:200, 1])
    plt.scatter(z_sparse[200:400, 0], z_sparse[200:400, 1])
    plt.scatter(z_sparse[400:450, 0], z_sparse[400:450, 1])
    plt.scatter(z_sparse[450:500, 0], z_sparse[450:500, 1])
    plt.savefig('../figs/sparse_pca_rep' + rep + '.pdf')
    plt.close()

    plt.scatter(z005[0:200, 0], z005[0:200, 1])
    plt.scatter(z005[200:400, 0], z005[200:400, 1])
    plt.scatter(z005[400:450, 0], z005[400:450, 1])
    plt.scatter(z005[450:500, 0], z005[450:500, 1])
    plt.savefig('../figs/naive_0.005_pca_rep' + rep + '.pdf')
    plt.close()

    plt.scatter(z01[0:200, 0], z01[0:200, 1])
    plt.scatter(z01[200:400, 0], z01[200:400, 1])
    plt.scatter(z01[400:450, 0], z01[400:450, 1])
    plt.scatter(z01[450:500, 0], z01[450:500, 1])
    plt.savefig('../figs/naive_0.01_pca_rep' + rep + '.pdf')
    plt.close()

    plt.scatter(z05[0:200, 0], z05[0:200, 1])
    plt.scatter(z05[200:400, 0], z05[200:400, 1])
    plt.scatter(z05[400:450, 0], z05[400:450, 1])
    plt.scatter(z05[450:500, 0], z05[450:500, 1])
    plt.savefig('../figs/naive_0.05_pca_rep' + rep + '.pdf')

    plt.close()
    u, s, v = np.linalg.svd(data)
    plt.scatter(u[0:200, 0], u[0:200, 1])
    plt.scatter(u[200:400, 0], u[200:400, 1])
    plt.scatter(u[400:450, 0], u[400:450, 1])
    plt.scatter(u[450:500, 0], u[450:500, 1])
    plt.savefig('../figs/standard_pca_rep' + rep + '.pdf')
    plt.close()

    u, s, v = np.linalg.svd(data[:, keep])
    plt.scatter(u[0:200, 0], u[0:200, 1])
    plt.scatter(u[200:400, 0], u[200:400, 1])
    plt.scatter(u[400:450, 0], u[400:450, 1])
    plt.scatter(u[450:500, 0], u[450:500, 1])
    plt.savefig('../figs/standard_pca_oracle_rep' + rep + '.pdf')
    plt.close()


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for pc in [0, 1]:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print('Plotting Loadings for PC', pc + 1)
    for rep in ['1', '2', '3', '4', '5']:
        w005 = np.load('../data/vi_mu_w_0.005_rep' + rep + '.npy')
        z005 = np.load('../data/vi_mu_z_0.005_rep' + rep + '.npy')
        w01 = np.load('../data/vi_mu_w_0.01_rep' + rep + '.npy')
        z01 = np.load('../data/vi_mu_z_0.01_rep' + rep + '.npy')
        w05 = np.load('../data/vi_mu_w_0.05_rep' + rep + '.npy')
        z05 = np.load('../data/vi_mu_z_0.05_rep' + rep + '.npy')
        keep = np.load('../data/keep_rep' + rep + '.npy')
        data = np.load('../data/data_rep' + rep + '.npy')
        w_sparse = np.load('../data/vi_mu_w_sparse_rep' + rep + '.npy')
        w_sparse_sigma = np.load('../data/vi_sigma_w_sparse_rep'
                                 + rep + '.npy')
        z_sparse = np.load('../data/vi_mu_z_sparse_rep' + rep + '.npy')
        psi_sparse = np.load('../data/vi_psi_0_sparse_rep' + rep + '.npy')

        u, s, v = np.linalg.svd(data)

        ys = (np.arange(len(v[pc])+1)) / len(v[pc])
        for idx, loadings in enumerate([v[pc],
                                        w005[:, pc],
                                        w01[:, pc],
                                        w05[:, pc],
                                        (1-psi_sparse[:, pc])
                                        * w_sparse[:, pc]]):
            loadings /= np.sqrt((loadings**2).sum())
            xs = [1e-100] + np.sort(np.abs(loadings)).tolist()
            line, = ax.plot(xs, ys, linewidth=1, color=colors[idx])

    ax.set_xscale('log')
    plt.xlim(1e-8, 1e-1)
    plt.savefig('../figs/loadings_pc' + str(pc+1) + '_all_reps.pdf')
    plt.close()
