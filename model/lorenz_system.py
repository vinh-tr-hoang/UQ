"""
Lorenz system  
"""
from scipy.integrate import odeint
from scipy.stats import norm

class lorenz_system_63:
    def __init__(self, rho = None, sigma = None, beta = None):
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.N = 3
        self.x0 = norm.rvs(size = self.N).reshape((self.N,))*4.
    def f(self,state, t):
      x, y, z = state  # unpack the state vector
      return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  # derivatives

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt
	
    ls = lorenz_system()
    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, 40.0, 0.01)
    states = odeint(ls.f, state0, t)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(states[:,0], states[:,1], states[:,2])
    plt.show()
