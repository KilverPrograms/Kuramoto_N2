from __future__ import division
import os
import h5py
import odespy
import matplotlib.pyplot as plt
import numpy as np


class Kuramoto:
    def __init__(self, omega, kappa):
        self.omega = omega
        self.kappa = kappa
        self.n_osc = len(omega)
        self.ics = 2 * np.pi * np.random.rand(self.n_osc)

    def f(self, u, t):
        kappa = self.kappa
        omega = self.omega
        n_osc = self.n_osc
        K = kappa / n_osc
        Theta = np.zeros((n_osc, n_osc))
        for i in xrange(n_osc):
            Theta[i, :] = u - u[i]
        return omega + np.sum(K * np.sin(Theta), axis=1)


class Solver:
    def __init__(self, problem, T, dt, method="RK4"):
        self.problem = problem
        self.T = T
        self.dt = dt
        self.method_class = eval('odespy.' + method)
        self.N = int(round(self.T / dt))

    def solve(self):
        self.solver = self.method_class(self.problem.f)
        self.solver.set_initial_condition(self.problem.ics)
        time_points = np.linspace(0., self.T, self.N + 1)
        self.u, self.t = self.solver.solve(time_points)
        self.du = np.gradient(self.u, self.dt, axis=0)
        print 'Final u(t={0:4.2f})={1}'.format(self.t[-1], np.array2string(self.u[-1, :]
                                                                           , precision=2, separator=', ',
                                                                           suppress_small=True))

    def save(self, fmt):
        if fmt == 'txt':
            np.savetxt("pase_" + str(self.problem.kappa) + ".txt", self.u)
        elif fmt == 'hdf5':
            if os.path.isfile("./phases.h5") == True:
                print "File exist!"
                phases = h5py.File("phases.h5", 'a')
                phases.create_dataset(str(self.problem.kappa), data=self.u)
            else:
                phases = h5py.File("phases.h5", 'w')
                phases.create_dataset(str(self.problem.kappa), data=self.u)