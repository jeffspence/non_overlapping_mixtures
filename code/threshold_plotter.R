# This plots Figure 3 in the paper -- the thresholding phenomenon.
# Text annotations were added in inkscape (https://inkscape.org)

vi_sigma = function(psi, psi_m1, sigma_0, sigma_1){
	return(1 / (1 + psi/sigma_0 + psi_m1/sigma_1))
}

vi_mu = function(psi, psi_m1, beta_hat, sigma_0, sigma_1){
	return(beta_hat * vi_sigma(psi, psi_m1, sigma_0, sigma_1))
}

elbo = function(log_psi, beta_hat, sigma_0, sigma_1, p_0){
	psi = 1 / (1 + exp(log_psi))
	psim1 = exp(log_psi) / (1 + exp(log_psi))
	vi_s = vi_sigma(psi, psim1, sigma_0, sigma_1)
	vi_m = vi_mu(psi, psim1, beta_hat, sigma_0, sigma_1)
	elbo = -0.5 * (vi_m^2 + vi_s - 2 * beta_hat * vi_m) + 0.5 * log(vi_s) - psi * log(psi) - psim1*log(psim1) - 0.5 * psi*log(sigma_0) - 0.5 * psim1*log(sigma_1) - (0.5 * psi/sigma_0 + 0.5 * psim1/sigma_1) * (vi_m^2 + vi_s) + psi * log(p_0) + psim1 * log(1-p_0)
	return(elbo)
}

vi_mean = function(beta_hat, sigma_0, sigma_1, p_0){
	to_return = c()
	for(this_sigma in sigma_0){
		obj = function(psi){
			return(elbo(psi, beta_hat, this_sigma, sigma_1, p_0))
		}
		log_best_psi = optimize(obj, interval = c(-500, 500), maximum=TRUE)$maximum
		best_psi = 1 / (1 + exp(log_best_psi))
		best_psim1 = exp(log_best_psi) / (1 + exp(log_best_psi))
		to_return = c(to_return, vi_mu(best_psi, best_psim1, beta_hat, this_sigma, sigma_1))
	}
	return(to_return)
}

true_post_0 = function(beta_hat, sigma_0, sigma_1, p_0){
	log_p0 = -0.5 * log(1 + sigma_0) + log(p_0) + 0.5 * beta_hat^2 * sigma_0 / (sigma_0 + 1) 
	log_p1 = -0.5 * log(1 + sigma_1) + log(1-p_0) + 0.5 * beta_hat^2 * sigma_1 / (sigma_1 + 1)
	max_logs = pmax(log_p0, log_p1) 
	post_0 = exp(log_p0 - max_logs)
	post_1 = exp(log_p1 - max_logs)
	normalizer = post_0 + post_1
	return(post_0 / normalizer)
}

true_mean = function(beta_hat, sigma_0, sigma_1, p_0){
	post_0 = true_post_0(beta_hat, sigma_0, sigma_1, p_0)
	post_1 = 1 - post_0
	return(beta_hat * (sigma_0 / (1 + sigma_0) * post_0 + sigma_1 / (1 + sigma_1) * post_1))
}



#Plot of thresholding phenomenon
prob_0 = 0.99
sigma_0 = 1e-10
sigma_1 = 1
betas = seq(0, 8, 0.05)
library(lattice)
to_plot = matrix(data=NA, nrow=length(sigma_0), ncol=length(betas))

for(i in 1:length(betas)){
	to_plot[, i] = vi_mean(betas[i], sigma_0, sigma_1, prob_0) # / true_mean(betas[i], 0, sigma_1, prob_0)
}

plot(0, type='n', xlim=c(0,8), ylim=c(0,4), xlab='Beta Hat', ylab='Posterior Mean')
lines(betas, to_plot[1,], lwd=3)
lines(betas, true_mean(betas, sigma_1, sigma_1, 0.5), col='red', lwd=3, lty=3)
lines(betas, true_mean(betas, 0, sigma_1, prob_0), col='blue', lwd=3, lty=2)