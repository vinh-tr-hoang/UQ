#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Lorenz system 96
"""

from scipy.integrate import odeint
from scipy.stats import norm

class lorenz_system:
    def __init__(self, N = 40, F = 8):
        self.N= N
        self.F = F
        self.x0 = norm.rvs(size = N).reshape((N,))  # initial state (equilibrium)
    def f(self,x, t):
          # compute state derivatives
          N = self.N
          d = np.zeros(N)
          # first the 3 edge cases: i=1,2,N
          d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
          d[1] = (x[2] - x[N-1]) * x[0]- x[1]
          d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
          # then the general case
          for i in range(2, N-1):
              d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
          # add the forcing term
          d = d + self.F
        
          # return the state derivatives
          return d

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    ls = lorenz_system()
    x0 = ls.x0;
    t = np.arange(5., 30.0, 0.01)
    
    x = odeint(ls.f, x0, t)
    
    # plot first three variables
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,2])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    plt.show()
