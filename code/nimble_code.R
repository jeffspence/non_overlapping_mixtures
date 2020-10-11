library(nimble, warn.conflicts = FALSE)
library(reticulate)

np <- import('numpy')

args <- commandArgs(trailingOnly = TRUE)
print('iteration:')
print(args[1])
print('sigma_sq_e:')
print(args[2])
niter <- 1000
sigma_sq_e <- args[2]
true_betas <- np$load(paste(c('../data/ldpred/true_betas_', sigma_sq_e, '.npy'), sep='', collapse=''))
beta_hats <- np$load(paste(c('../data/ldpred/beta_hats_', sigma_sq_e, '.npy'), sep='', collapse=''))
ld_mats <- np$load(paste(c('../data/ldpred/ld_mats_', sigma_sq_e, '.npy'), sep='', collapse=''))
sigma_sq_e <- as.numeric(args[2])

super_iter <- strtoi(args[1])
model_code <- nimbleCode({
	for(i in 1:N){
		beta_latent[i] ~ dnorm(0, 1)
		z[i] ~ dbern(p_causal)
		beta[i] <- beta_latent[i] * z[i]
	}
	mean_vec[1:N] <- cov[1:N, 1:N] %*% beta[1:N]
	cov_mat[1:N, 1:N] <- cov[1:N, 1:N]*sigma_sq_e
	beta_hat[1:N] ~ dmnorm(mean_vec[1:N], cov=cov_mat[1:N, 1:N])
})
	
	
cov <- ld_mats[super_iter, ,]
beta_hat <- beta_hats[super_iter, ]
model_consts <- list(N = length(beta_hat), p_causal = 0.01, cov=cov, sigma_sq_e= sigma_sq_e)
model_data <- list(beta_hat = beta_hat)
model_inits <- list(beta_latent = rep(0, length(beta_hat)), z = rep(0, length(beta_hat)))
model <- nimbleModel(code=model_code,
		     name='PRS model',
		     constants=model_consts,
		     data=model_data,
		     inits=model_inits)
Cmodel <- compileNimble(model)
model_conf <- configureMCMC(model, print=TRUE)
model_conf$addMonitors(c('beta'))
model_mcmc <- buildMCMC(model_conf)
Cmodel_mcmc <- compileNimble(model_mcmc, project=model)
	
set.seed(42)
Cmodel_mcmc$run(niter)
samples <- as.matrix(Cmodel_mcmc$mvSamples)
beta_means <- colMeans(samples[1:niter, 1:length(beta_hat)])
print('CORRELATION:')
print(cor(beta_means, true_betas[super_iter, ]))
print('MSE:')
print(mean((beta_means - true_betas[super_iter, ])^2))
write(c('CORRELATION:', cor(beta_means, true_betas[super_iter, ]),
	'MSE:', mean((beta_means - true_betas[super_iter, ])^2)),
      file=paste(c('../data/ldpred/nimble_results_', args[1], '_', args[2], '.txt'),
                 sep='', collapse=''))
