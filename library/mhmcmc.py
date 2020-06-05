from scipy.stats import uniform
import numpy as np
class mhmcmc:
    def __init__(self,_func_prior_pdf, _func_prior_gen, _func_likelihood,_func_model_,
                 _func_kernel_gen, n_MC, dim,**kwargs):
        self.n_MC = n_MC
        self.dim  = dim
        self.x_MCMC = np.zeros (shape = (n_MC, dim))       
        self.prior_pdf = _func_prior_pdf
        self.prior_gen = _func_prior_gen
        self.likelihood = _func_likelihood
        self.forward_model = _func_model_
        self.kernel_gen = _func_kernel_gen
        if 'loglikelihood' in kwargs:
            self.loglikelihood = kwargs['loglikelihood']
        else:
            self.loglikelihood = False
        if '_func_kernelRatio' in kwargs:
            self.kernelRatio = kwargs['_func_kernelRatio']
        else:
            self.kernelRatio = self.unitRatio
    def unitRatio(self, xi,xj):
        return 1.                      
    def run_MCMC(self, **kwargs): # kwargs: keep_rejected_point
        
        if 'x_int' in kwargs:
            self.x_MCMC[0,:] = kwargs['x_int']
        else:
            x = self.prior_gen()
            self.x_MCMC[0,:] = x 
            print('initial value: ', self.x_MCMC[0,:])
        y = self.forward_model(x)
        self.y_MCMC = np.zeros (shape = (self.n_MC, y.size))
        self.y_MCMC[0,:] = y
        pdf_x_prev = self.prior_pdf (x)
        pdf_epsilon_prev =  self.likelihood (y)
        pdf_x_curr = 0.
        pdf_epsilon_curr = 0.
        number_repeated_sample = 0
        if 'keep_rejected_point' in kwargs:
            self.x_computed = np.zeros (shape = (self.n_MC, self.dim))
            self.x_computed [0,:] = x
            self.y_computed = np.zeros (shape = (self.n_MC, y.size))
            self.y_computed [0,:] = y
        for i in range (1,self.n_MC):                              
            if ((number_repeated_sample >= 100)): # start the chain again if no update for 100 steps                     
                print ("MCMC is repeated form step ", i-number_repeated_sample, ' to step ', i, 
                       " with value: ", self.x_MCMC[i-1,:])
                print (pdf_x_curr, pdf_epsilon_curr, successful_update, ratio, x)
                x = self.prior_gen ()
                y = self.forward_model(x)
                if self.loglikelihood:
                    while self.likelihood(y) < - 1e15:
                        x = self.prior_gen ()
                        y = self.forward_model(x)
                else:
                    while self.likelihood(y) < 1.e-15:
                        x = self.prior_gen () 
                        y = self.forward_model(x)
                   
                number_repeated_sample = 0  
            if ((number_repeated_sample < 100)): # Generate new samples 
                x = self.kernel_gen(self.x_MCMC[i-1,:])                                
                y = self.forward_model(x)
            if 'keep_rejected_point' in kwargs:
                self.x_computed[i] = x 
                self.y_computed[i] = y 
            pdf_x_curr = self.prior_pdf (x)
            pdf_epsilon_curr = self.likelihood (y)
            
            # Acept or refuse new sample
            temp = uniform.rvs(size = 1)
            if self.loglikelihood:
                ratio = (pdf_x_curr + pdf_epsilon_curr) - (pdf_x_prev + pdf_epsilon_prev)
                successful_update = (np.log(temp) <= ratio) 
            else:
                if (pdf_x_prev*pdf_epsilon_prev):
                    kernelRatio = self.kernelRatio(self.x_MCMC[i-1,:],x)
                    ratio = (pdf_x_curr*pdf_epsilon_curr)/(pdf_x_prev*pdf_epsilon_prev)
                    ratio = ratio*kernelRatio                     
                else:
                    ratio =1.                                    
                successful_update = (temp <= ratio) 
            if successful_update:
                self.x_MCMC[i,:] = x
                self.y_MCMC[i,:] = y
                number_repeated_sample = 0
                pdf_x_prev = pdf_x_curr
                pdf_epsilon_prev = pdf_epsilon_curr
            else:
                self.x_MCMC [i,:] = self.x_MCMC[i-1,:]
                self.y_MCMC [i,:] = self.y_MCMC[i-1,:]
                number_repeated_sample = number_repeated_sample +1
            if i%500 == 0:
                print ("MCMC current step:", i )     
                print ("mean value", [self.x_MCMC[0:i,j].mean() for j in range(0,self.dim)] )
###############################################################################                
## Ouput 
###############################################################################
        if self.dim == 1:
            self.x_MCMC = self.x_MCMC.reshape((self.x_MCMC.size,))
            self.y_MCMC = self.y_MCMC.reshape((self.y_MCMC.size,))
        if 'keep_rejected_point' in kwargs:
            return self.x_MCMC, self.x_computed #, self.std_converge_flag  
        else: 
            return self.x_MCMC,
