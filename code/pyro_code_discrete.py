import torch
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from torch.distributions import constraints
import numpy as np
import sys
from time import time
from pyro.poutine import block, replay, trace
from functools import partial


LR = 1e-3
NUM_PARTICLES = 2
TOTAL_ITS = 100000
GLOBAL_K = 50
N = 1000
p_causal = 0.01
GENETIC_MEAN = torch.tensor(np.zeros(N))
GENETIC_SD = torch.tensor(np.ones(N))
GENETIC_MIX = torch.tensor(np.log([1-p_causal, p_causal]))
sigma_sq_e = float(sys.argv[1])


def prs_model(beta_hat, obs_error):
    z = pyro.sample(
        'z',
        dist.Independent(dist.Bernoulli(torch.tensor([p_causal]*N)), 1)
    )
    beta = pyro.sample(
        'beta_latent',
        dist.Independent(dist.Normal(GENETIC_MEAN,
                                     GENETIC_SD), 1)
    )
    beta_hat = pyro.sample(
        'beta_hat',
        dist.MultivariateNormal(torch.mv(obs_error, beta*z),
                                covariance_matrix=obs_error*sigma_sq_e),
        obs=beta_hat
    )
    return beta_hat


def prs_guide(index):
    psi_causal = pyro.param(
        'var_psi_causal_{}'.format(index),
        torch.tensor(np.ones(N)*p_causal),
        constraint=constraints.unit_interval
    )
    z = pyro.sample(
        'z',
        dist.Independent(dist.Bernoulli(psi_causal), 1)
    )
    means = pyro.param(
        'var_mean_{}'.format(index),
        torch.tensor(np.zeros(N))
    )
    scales = pyro.param(
        'var_scale_{}'.format(index),
        torch.tensor(np.ones(N)),
        constraint=constraints.positive
    )
    beta_latent = pyro.sample(
        'beta_latent',
        dist.Independent(dist.Normal(means, scales), 1)
    )
    return z, beta_latent


def approximation(components, weights):
    assignment = pyro.sample('assignment', dist.Categorical(weights))
    results = components[assignment]()
    return results


def relbo(model, guide, *args, **kwargs):
    approximation = kwargs.pop('approximation')
    traced_guide = trace(guide)
    elbo = pyro.infer.Trace_ELBO(num_particles=NUM_PARTICLES)
    loss_fn = elbo.differentiable_loss(model, traced_guide, *args, **kwargs)
    guide_trace = traced_guide.trace
    replayed_approximation = trace(replay(block(approximation,
                                                expose=['beta_latent', 'z']),
                                          guide_trace))
    approximation_trace = replayed_approximation.get_trace(*args, **kwargs)
    relbo = -loss_fn - approximation_trace.log_prob_sum()
    return -relbo


def run_svi(beta_hat, obs_error, K, true_beta):
    num_steps = TOTAL_ITS//K
    start = time()
    pyro.clear_param_store()
    pyro.enable_validation(True)

    def my_model():
        return prs_model(torch.tensor(beta_hat),
                         torch.tensor(obs_error))

    initial_approximation = partial(prs_guide, index=0)
    components = [initial_approximation]
    weights = torch.tensor([1.])
    wrapped_approximation = partial(approximation,
                                    components=components,
                                    weights=weights)
    optimizer = pyro.optim.Adam({'lr': LR})
    losses = []
    wrapped_guide = partial(prs_guide, index=0)
    svi = pyro.infer.SVI(
        my_model,
        wrapped_guide,
        optimizer,
        loss=pyro.infer.Trace_ELBO(num_particles=NUM_PARTICLES)
    )
    for step in range(num_steps):
        loss = svi.step()
        losses.append(loss)
        if step % 100 == 0:
            print('\t', step, np.mean(losses[-100:]))
        if step % 100 == 0:
            pstore = pyro.get_param_store()
            curr_mean = pstore.get_param(
                'var_mean_{}'.format(0)).detach().numpy()
            curr_psis = pstore.get_param(
                'var_psi_causal_{}'.format(0)).detach().numpy()
            curr_mean = curr_mean * curr_psis
            print('\t\t', np.corrcoef(true_beta, curr_mean)[0, 1],
                  np.mean((true_beta - curr_mean)**2))
    pstore = pyro.get_param_store()
    for t in range(1, K):
        print('Boost level', t)
        wrapped_guide = partial(prs_guide, index=t)
        losses = []
        optimizer = pyro.optim.Adam({'lr': LR})

        svi = pyro.infer.SVI(my_model, wrapped_guide, optimizer, loss=relbo)
        new_weight = 2 / ((t+1) + 2)
        new_weights = torch.cat((weights * (1-new_weight),
                                 torch.tensor([new_weight])))
        for step in range(num_steps):
            loss = svi.step(approximation=wrapped_approximation)
            losses.append(loss)
            if step % 100 == 0:
                print('\t', step, np.mean(losses[-100:]))
            if step % 100 == 0:
                pstore = pyro.get_param_store()
                curr_means = [
                    pstore.get_param(
                        'var_mean_{}'.format(s)).detach().numpy()
                    for s in range(t+1)
                ]
                curr_psis = [
                    pstore.get_param(
                        'var_psi_causal_{}'.format(0)).detach().numpy()
                    for s in range(t+1)
                ]
                curr_means = np.array(curr_means) * np.array(curr_psis)
                curr_mean = new_weights.detach().numpy().dot(curr_means)
                print('\t\t', np.corrcoef(true_beta, curr_mean)[0, 1],
                      np.mean((true_beta - curr_mean)**2))

        components.append(wrapped_guide)
        weights = new_weights
        wrapped_approximation = partial(approximation,
                                        components=components,
                                        weights=weights)
        # scales.append(
        #     pstore.get_param('var_mean_{}'.format(t)).detach().numpy()
        # )
    print('BBBVI ran in', time() - start)
    pstore = pyro.get_param_store()
    curr_means = [
        pstore.get_param(
            'var_mean_{}'.format(s)).detach().numpy()
        for s in range(K)
    ]
    return weights.detach().numpy().dot(np.array(np.array(curr_means)))


if __name__ == '__main__':
    beta_hats = np.load('../data/ldpred/beta_hats_' + str(sigma_sq_e) + '.npy')
    cov_mats = np.load('../data/ldpred/ld_mats_' + str(sigma_sq_e) + '.npy')
    true_betas = np.load('../data/ldpred/true_betas_'
                         + str(sigma_sq_e) + '.npy')

    cors = []
    mse = []
    for i in [int(sys.argv[2])-1]:
        print((np.abs(true_betas[i, 0:N]) > 1e-10).sum(), 'num nonzero')
        print(np.mean(true_betas[i, 0:N]**2), 'null MSE')
        post_mean = run_svi(beta_hats[i][0:N],
                            cov_mats[i][0:N, 0:N],
                            GLOBAL_K,
                            true_betas[i, 0:N])
        cors.append(np.corrcoef(post_mean, true_betas[i, 0:N])[0, 1])
        mse.append(np.mean((post_mean - true_betas[i, 0:N])**2))
    np.savetxt('../data/ldpred/pyro_discrete_' + str(sigma_sq_e)
               + '_rep_' + sys.argv[2] + '_cor.txt', cors)
    np.savetxt('../data/ldpred/pyro_discrete_' + str(sigma_sq_e)
               + '_rep_' + sys.argv[2] + '_mse.txt', mse)
