import numpy as np
from scipy.stats import norm, gaussian_kde, multivariate_normal
import matplotlib.pyplot as plt
from library.mhmcmc import mhmcmc


def forward (q):
    return np.array ([q[0]**3+ q[1], q[1]])

prior_mean = np.array([0., 0.])
prior_std = np.array([2., 2.])
measurement_error_dist = multivariate_normal(mean = [0., 0], cov = np.eye(2)*0.05)
prior_dist = multivariate_normal (mean = prior_mean, cov = np.diag(prior_std))
q_exact = np.array([1., 1.])
hat_y = forward(q_exact) + measurement_error_dist.rvs()

def kernel_generator (q):
    return q + multivariate_normal.rvs(mean = prior_mean, cov = np.diag(prior_std))/3.

def likelihood (y):
    return measurement_error_dist.pdf(y - hat_y)

np.random.seed(2)
n_MCMC = 20000
mcmcalgorithm = mhmcmc(prior_dist.pdf, prior_dist.rvs, 
                    likelihood, forward,
                    kernel_generator, n_MCMC, dim = 2, loglikelihood = False)
mcmcalgorithm.run_MCMC(keep_rejected_point = 0)

qposterior = mcmcalgorithm.x_MCMC
# Visualization 
for j in range (0, 2):
    print("   ")
    print("posterior mean ",j ,'th dimesion', qposterior[:,j].mean())
    print("posterior std ",j ,'th dimesion', np.std(qposterior[:,j]))
    print("   ")




###################################################################
#Visualisation 
for j in range (0, 2):
    q_grid = np.linspace(prior_mean[j]-5*prior_std[j]
            , prior_mean[j]+5*prior_std[j], 1000)
    posterior_pdfx = gaussian_kde(qposterior[:,j]).evaluate(q_grid)
    priorj = norm (loc = prior_mean[j], scale = prior_std[j])
    prior_pdfx = np.array([priorj.pdf(x)  for x in q_grid])
        
    
    
    plt.figure('density of component ' + str(j+1))
    plt.plot(q_grid,posterior_pdfx, label ='posterior')
    plt.plot(q_grid, prior_pdfx, label ='prior')
    plt.plot(q_exact[j], 0,'s', label ='exact')
    plt.legend()
    plt.xlabel('q')
    plt.ylabel('pdf')
    plt.show()

