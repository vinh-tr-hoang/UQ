"""
surrogate model using generalized polynomial chaos expansion
"""
import numpy as np
from scipy.stats import uniform
import os,sys
my_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(my_path+'/../library')
from gpce import onedgpce

var = uniform(loc = -4., scale =  8.)
dist_type = 'norm'
order = 15
import os,sys
my_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(my_path+'/../library')

pceModel = onedgpce(dim = 1, var = var, order = order, dist_type = dist_type)
print ("check orthonality: oder4, oder 4", pceModel.check_orthogonal(4, 4))    
print ("check orthonality via Quad: oder4, oder 4", pceModel.check_orthogonal_viaQuad(4, 4)) 
print ("check orthonality: oder2, oder 4", pceModel.check_orthogonal(2, 4))
print ("check orthonality via Quad: oder2, oder 4", pceModel.check_orthogonal_viaQuad(2, 4)) 

def forward_model(x):
    return  np.sin(x)
def forward_model_logx(logx):
    return  forward_model(np.exp(logx))
#pceModel.get_modelcoef(forward_model, method = "quad", nbint = order +1)
pceModel.get_modelcoef(forward_model)        
xssamples = np.sort(var.rvs(size = 1000))
import matplotlib.pyplot as plt
plt.plot(xssamples, forward_model(xssamples), label = 'forward function' )
plt.plot(xssamples, pceModel.pceModeleval(xssamples),'--', label = 'pce approximation' )
plt.legend()
plt.show()
    