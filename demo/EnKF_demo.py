"""
Ensemble Kalman filter for tracking Lorenz 63 systems 
Author: Truong-Vinh Hoang
"""
import numpy as np
from scipy.stats import multivariate_normal, norm
#from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from lorenz_system import lorenz_system_63 as lorenz_system
from scipy.integrate import odeint
import os,sys
my_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(my_path+'/../model')

# Lorenz system 
ls = lorenz_system()
dim_state = ls.N
state0 = ls.x0 # initial state


options = {
"N" : 200, # size of the ensembles
"dimy" : 3,
"deltat" : 0.1, # time step for solving ode
"deltat_obs" : 0.3, # observation time step
"is_obsmap_linear" : True,
"is_laplace" : False,
"seed" : 101,
"std_noise" : 2.,
"delta_anti_generacy" : 1.0,
"numberneural" : 10,
"max_assimilation_step" : int (100)
}
t = np.arange(0., 100, options["deltat"]) 
true_state = odeint(ls.f, state0, t)
np.random.seed(seed = options["seed"])
# For advance version:
max_assimilation_step = int(options["max_assimilation_step"])
# For advance version:
t = t + norm.rvs(size= t.size).reshape(t.shape)*1e-4
if options["deltat_obs"] == 0.1:
    options["delta_anti_generacy"] = 1.01
#############################    Noisy observation    #########################
def observation_map(x):
    if options["dimy"] == dim_state and (1-options["is_obsmap_linear"]):
        noise = norm.rvs(size = x.shape[0]*options["dimy"])*options["std_noise"]
        return x + x**3/15**2. + noise.reshape(x.shape)
    if options["dimy"] == dim_state and options["is_obsmap_linear"] and (1-options["is_laplace"]):
        noise = norm.rvs(size = x.shape[0]*options["dimy"])*options["std_noise"]
        return x  + noise.reshape(x.shape)
    if options["is_laplace"] and options["is_obsmap_linear"]:
        noise = np.random.laplace(size = x.shape[0]*options["dimy"])*options["std_noise"]
        return x  + noise.reshape(x.shape)  


obs_step = (np.arange(0, t.size-1, options["deltat_obs"]/options["deltat"]))
obs_step = obs_step.astype(int)
obs_step = obs_step[:max_assimilation_step]
####################
obs_noised = observation_map(true_state[obs_step])

t =t [:obs_step[max_assimilation_step-1]+1]
true_state =true_state [:obs_step[max_assimilation_step-1]+1,:]
obs_noised = obs_noised [:obs_step[max_assimilation_step-1]+1,:]

##########################  Performance measures    ##########################
def RMSE(predict, true, predict_std, dim_state = dim_state): # Average root mean square error
    error = predict -true
    #print('error = ', error)
    error_L2norm = np.sqrt(np.mean (error**2, axis = 1))
    #print('error_L2norm = ', error_L2norm)
    averaged_error = np.mean(error_L2norm[1:])
    print('RMSE = ', averaged_error)
    averaged_error_rel = np.mean(error_L2norm[1:])/np.sqrt(np.var(true[:]))
    print('relative average RMSE = ', averaged_error_rel)
    average_spread = np.sum(np.mean(predict_std, axis = 1))/dim_state
    average_spread_rel = average_spread /np.sqrt(np.var(true[:]))
    print('averaged spread = ', average_spread)
    print('relative averaged spread  = ', average_spread_rel)
    return averaged_error, averaged_error_rel, average_spread, average_spread_rel

#################################### Asimilation ##############################
# prior state 0 
mean = np.zeros((dim_state,))
cov = np.eye(dim_state)*10.
st0 = multivariate_normal(mean, cov).rvs(options["N"])

EnFK_assimilated_state = np.zeros(shape = (options["N"], dim_state, obs_step.size))
EnFK_predicted_state = np.zeros(shape = (options["N"], dim_state, t.size))
EnFK_assimilated_state[:,:,0] =st0
for i in range (1,obs_step.size ):
    prediction = np.zeros(shape = (options["N"],dim_state))
    t_i =t[obs_step[i-1]:obs_step[i]+1]
    for samples in range (0,options["N"]):         
        temp = odeint(ls.f, EnFK_assimilated_state[samples,:,i-1], t_i)
        prediction[samples,:]  = temp [-1,:]
        EnFK_predicted_state [samples,:,obs_step[i-1]:obs_step[i]+1] =temp.transpose()
    predicted_obs = observation_map(prediction)
    C_xfyf = (np.matmul((prediction - prediction.mean(axis = 0)  ).transpose(),
                       ( predicted_obs- predicted_obs.mean(axis = 0) ))/options["N"])
    C_yfyf = (np.matmul((predicted_obs- predicted_obs.mean(axis = 0) ).transpose(),
                       ( predicted_obs- predicted_obs.mean(axis = 0) ))/options["N"])
    K_gain = np.matmul(C_xfyf,np.linalg.inv(C_yfyf))
    EnFK_assimilated_state[:,:,i] = (prediction 
                                    + np.matmul(K_gain, (obs_noised[i,:]-predicted_obs).transpose()).transpose())
    tempmean = EnFK_assimilated_state[:,:,i].mean(axis = 0)
    EnFK_assimilated_state[:,:,i] = tempmean + options["delta_anti_generacy"]*(EnFK_assimilated_state[:,:,i] -tempmean) 
    if i% 10 == 0:
        print ("ENKF step "+ str(i) + ' of  '+ str(obs_step.size) + 'total steps ')  

errorEnKF =RMSE( np.mean(EnFK_assimilated_state[:,:, int(obs_step.size/2):], axis= 0) , 
                true_state[obs_step[int(obs_step.size/2):],:].transpose(),
                  np.std(EnFK_assimilated_state[:,:, int(obs_step.size/2):], axis= 0))

########################      Visualisation    ################################
plt.figure()
if options["is_obsmap_linear"]:
    plt.plot(t[obs_step],obs_noised[:,0], 's' , label ='noisy observation')

plt.plot(t,true_state[:,0], '-r' , label =' true process')
plt.plot(t[obs_step],np.mean(EnFK_assimilated_state[:,0,:], axis= 0), '-' , label = 'EnKF mean' )
plt.xlabel('t', fontsize=16, color='black') 
plt.ylabel('q[1]', fontsize=16, color='black') 
plt.legend(loc = 'top left')
