"""
Generalized polynomial chaos expansion
Author: Truong-Vinh Hoang
"""
import numpy as np
from scipy.stats import norm, uniform, gaussian_kde
import scipy.special as sp
from numpy import poly1d

class onedgpce():
    def __init__(self,dim = 1, dist_type = None, var = None, order = None):
        self.dim = dim
        self.order = order
        self.nbpoly = order + 1
        self.var = var
        self.dist_type = dist_type
        #self.coef = np.zeros(self.nbpoly, self.order + 1)
        if dist_type =='norm':
            self.funcs = np.array([sp.hermitenorm(i)for i in range (0, self.order+1)])            
            for i in range(0, self.order+1):
                normalized_const = np.sqrt(sp.factorial(i))
                self.funcs[i] = poly1d(self.funcs[i].c/normalized_const)
        if dist_type =='uniform':
            #The polynomials  are orthogonal over [0,1] with weight function 1.
            self.funcs = np.array([sp.sh_legendre(i)for i in range (0, self.order+1)])
            for i in range(0, self.order+1):
                normalized_const = np.sqrt(1./(2.*i+1.))
                self.funcs[i] = poly1d(self.funcs[i].c/normalized_const)                            
        self.modelcoef = np.zeros((self.nbpoly,))
    def evalfuncs(self,order,x):
        xs = self.scaling(x)
        return self.funcs[order](xs)
    def pceModeleval(self,x):
        xs = self.scaling(x)
        pcemodel = 0.
        for i in range(0, self.nbpoly):
            pcemodel = pcemodel +self.modelcoef[i]*self.funcs[i](xs)
        return pcemodel
    def scaling (self,x):
        if self.dist_type == 'norm':
            xs = (x - self.var.mean())/self.var.std()
        if self.dist_type == 'uniform':
            xs = (x - self.var.low)/(self.var.high-self.var.low) # in [0,1]            
        return xs
    def descaling (self,xs):
        if self.var.name == 'norm':
            x = (xs*self.var.std() + self.var.mean())
        if self.var.name == 'uniform':
            x = (xs)*(self.var.high-self.var.low) +  self.var.low            
        return x  
    def check_orthogonal(self, orderi, orderj):
        x = self.var.rvs(size =100000)
        xs = self.scaling(x)
        yi = self.funcs[orderi](xs)
        yj = self.funcs[orderj](xs)
        y = yi*yj
        return np.mean (y)
    def check_orthogonal_viaQuad(self, orderi, orderj):
        xs, wxs = self.scaled_quadrature_points()
        yi = self.funcs[orderi](xs)
        yj = self.funcs[orderj](xs)
        y = yi*yj*wxs
        return np.sum (y)
    def scaled_quadrature_points(self):
        if self.dist_type == 'norm':
            xg, wg = sp.roots_hermitenorm(self.order) # weight function exp(-x^2/2)
            wg = wg/ np.sqrt(2*np.pi) # for standard normal distribution
        if self.dist_type == "uniform":
            xg, wg = sp.roots_sh_legendre(self.order) # weight function exp(-x^2/2)
            #wg = wg.# for standard normal distribution            
        return xg, wg
    def get_modelcoef(self, forward_model, **kwargs):
        print (kwargs)
        if 'method' in kwargs:
            method = kwargs['method']
            nbint = kwargs['nbint']
        else:
            method = 'MC'
            nbint = 20000
        print("projection using "+method, " method")
        x,xs,wx = self.intergationPoints(method, nbint)            
        y = forward_model(x)
        for i in range(0, self.nbpoly):
            funci = self.funcs[i](xs)
            # Projections
            self.modelcoef[i] = np.sum (y*funci*wx)

    
    def intergationPoints(self,method,nbint):
        if method == 'quad':
            if self.dist_type == 'norm':
                xs,wxs = sp.roots_hermitenorm(nbint)
                wx = wxs /np.sqrt(2*np.pi)
                x = self.descaling(xs)
            if self.dist_typee == 'uniform':
                xs,wxs = sp.roots_sh_legendre(nbint)
                wx = wxs
                x = self.descaling(xs)
        if method =='MC':
            x = self.var.rvs(size = nbint)
            wx = 1./x.size
            xs = self.scaling(x)
        return x, xs,wx

if __name__ == '__main__':
    var = uniform(loc = -4., scale =  8.)
    dist_type = 'norm'
    order = 15

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
    
    
    