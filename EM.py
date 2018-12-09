import numpy as np 

from scipy.stats import multivariate_normal
class EM:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters
        self.epochs = 10000
        np.random.seed(1904)


    def initialize_mu(self):
        mu = self.data[np.random.choice(self.data.shape[0], self.num_clusters, replace=False)]
        
        return mu

    def initialize_sigma(self):
        D = self.data.shape[1]
        sigma = np.zeros((self.num_clusters, D, D))

        for k in range(self.num_clusters):
            for d in range(D):
                sigma[k, d, d] = 10000

        return sigma 

    def initialize_pi(self):
        pi = np.full(self.num_clusters, 1/self.num_clusters)

        return pi 

    def initialize_params(self):
        return (self.initialize_mu(), self.initialize_sigma(), self.initialize_pi())

    def eval_log_likelihood(self, mu, sigma, pi):
        K = pi.shape[0]
        N = self.data.shape[0]
        ln_p = 0
        for n in range(N):
            p_n = 0
            for k in range(K):
                p_n += pi[k] * multivariate_normal.pdf(self.data[n], mean = mu[k], cov = sigma[k])
        ln_p += np.log(p_n)

        return ln_p

    def E_step(self, mu, sigma, pi):
        K = pi.shape[0]
        N = self.data.shape[0]
        gamma_z = np.zeros((K,N))
        for k in range(K):
            gamma_z[k, :] = pi[k] * multivariate_normal.pdf(self.data, mean = mu[k], cov = sigma[k])

        gamma_z /= gamma_z.sum(0)

        return gamma_z

    def M_step(self, mu, sigma, gamma_z):
        mu_new = np.zeros(mu.shape)
        sigma_new = np.zeros(sigma.shape)
        pi_new = np.zeros(mu.shape[0])
        for k in range(gamma_z.shape[0]):
            N = self.data.shape[0]
            total_k = sum(gamma_z[k])
            gamma_zk = gamma_z[k].reshape(-1, 1)
            mu_new[k] = sum(gamma_zk * self.data)/total_k
            for n in range(N):
                sigma_new[k] += gamma_z[k, n] * np.dot((self.data[n, :] - mu_new[k]).reshape(-1,1), (self.data[n, :] - mu_new[k]).reshape(1, -1))
            sigma_new[k] /= total_k
            pi_new[k] = total_k/N

        return (mu_new, sigma_new, pi_new)

    def execute(self):
        (mu, sigma, pi) = self.initialize_params()
        prev_likelihood = self.eval_log_likelihood(mu, sigma, pi)
        for epoch in range(self.epochs):
            gamma_z = np.nan_to_num(self.E_step(mu, sigma, pi))
            (mu, sigma, pi) = self.M_step(mu, sigma, gamma_z)
            log_likelihood = self.eval_log_likelihood(mu, sigma, pi)
            if(abs(log_likelihood - prev_likelihood < 0.00001)):
                break
            prev_likelihood = log_likelihood
        
        return (mu, sigma, pi)

