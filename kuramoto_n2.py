#!/usr/local/bin/python2
from __future__ import division
import odespy
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

__version__ = "0.0.2"
__author__  = "Kilver J. Campos"

class Kuramoto_N2(object):
    """
    Implementation of Kuramoto coupling model of two oscillators.
    It uses NumPy and odespy implementation ODE solvers for
    numerical integration.
    
    Usage example:
    >>> init_values = { 'omega'  : [0.5, 1.5] ,
    >>>         'kappa'  : 0.99 ,
    >>>         'T'      : 100.0 ,
    >>>         'dt'     : 0.1 ,
    >>>         'method' : "Dopri5"}
    >>> problem = Kuramoto_N2(init_values)
    >>> problem.solve()
    >>> problem.plot()
    
    [1] Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
        (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3
    """
    
    def __init__(self, init_values):
        
        self.kappa  = init_values['kappa']
        self.omega  = np.array(init_values['omega'])
        self.T      = init_values['T']
        self.dt     = init_values['dt']
        self.method = init_values['method']
        
        self.n_osc  = len(self.omega)
        self.ics    = 2*np.pi*np.random.rand(self.n_osc)
        self.N      = int(round(self.T/self.dt))
        
    def f(self,u,t):
        kappa = self.kappa
        omega = self.omega
        return [omega[0] + kappa / 2 * np.sin (u[1] - u[0]),
                omega[1] + kappa / 2 * np.sin (u[0] - u[1])]
    
    
    def solve(self):
        self.method_class = eval('odespy.' + self.method)
        self.solver = self.method_class(self.f)
        self.solver.set_initial_condition(self.ics)
        time_points = np.linspace(0., self.T, self.N+1)
        self.u, self.t = self.solver.solve(time_points)
        print 'Final u(t={0:4.2f}) = {1}'.format(self.t[-1], np.array2string(self.u[-1,:], separator=', ' ,suppress_small=True))
    
    def plot(self):
        plt.close()
        plt.figure(figsize=(12,8))
        grid = plt.GridSpec(2,2)
        # ------ Oscillator Fase ------- #
        plt.subplot(grid[0,0])
        plt.plot(self.t, self.u)#%(2*np.pi))
        #plt.ylim(0,2*np.pi)
        plt.xlim(0,self.T)
        plt.title("Oscillators fase")
        plt.xlabel("Time")
        plt.ylabel(r"$\theta_1,\theta_2$")
        plt.legend([r"$\theta_1$",r"$\theta_2$"])
        # ------- Fase Difference ------- #
        plt.subplot(grid[0,1])
        plt.plot(self.t, self.u[:,0]-self.u[:,1])
        plt.xlim(0,self.T)
        plt.title("Fase Difference")
        plt.xlabel("Time")
        plt.ylabel(r"$|\theta_1-\theta_2|$")
        # ------- Order Parameter ------- #
        plt.subplot(grid[1,0])
        plt.plot(self.t, (np.cos(.5*(self.u[:,0]-self.u[:,1]))))
        plt.xlim(0,self.T)
        plt.ylim(-1.05,1.05)
        plt.title("Order Parameter")
        plt.xlabel("Time")
        plt.ylabel(r"$r$")
        # ------- Agular velociti ------- #
        self.du = np.gradient(self.u,self.dt,axis=0)
        plt.subplot(grid[1,1])
        plt.plot(self.t, self.du[:,0])
        plt.plot(self.t, self.du[:,1])
        plt.plot(self.t, np.gradient(self.u[:,0]-self.u[:,1],self.dt),'k')
        plt.xlim(0,self.T)
        plt.title("Aparent Angular Velocity")
        plt.xlabel("Time")
        plt.ylabel(r"$\Omega$")
        plt.legend([r"$\dot{\theta_1}$",r"$\dot{\theta_2}$",r"$\dot{\varphi}$"])

        plt.tight_layout()
        plt.savefig("time_parameters.pdf",)
        plt.show()

    def save(self,fmt):
        if fmt == 'txt':
            np.savetxt("pase_"+str(self.kappa)+".txt",self.u)
        elif fmt == 'hdf5':
            if os.path.isfile("./phases.h5")==True:
                print "File exist!"
                phases = h5py.File("phases.h5",'a')
                phases.create_dataset(str(self.kappa),data=self.u)
            else:
                phases = h5py.File("phases.h5", 'w')
                phases.create_dataset(str(self.kappa), data=self.u)

