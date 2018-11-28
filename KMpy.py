#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os, copy, h5py, odespy
import matplotlib.pyplot as plt
import numpy as np
from numba import jitclass


__author__ = "Kilver J. Campos"
__copyright__ = "Copyright 2018, Kuramoto Model Network on Python (pyKuNet)"
__credits__ = ["Kilver J. Campps"]
__license__ = "GPL3"
__version__ = "0.0.1"
__maintainer__ = "Kilver J. Campos"
__email__ = "kilvercampos@gmail.com"
__status__ = "Production"


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
        print('Final u(t={0:4.2f})={1}'.format(self.t[-1], np.array2string(self.u[-1, :]
                                                                           , precision=2, separator=', ',
                                                                           suppress_small=True)))

    def opr_t(self):
        self.r = np.abs((1 / float(self.problem.n_osc)) * np.sum(np.exp(1j * self.u), axis=1))

    def dot_u(self):
        self.du = np.gradient(self.u, self.dt, axis=0)

    def all(self):
        self.solve()
        self.dot_u()
        self.opr_t()

    def save(self, fmt):
        if fmt == 'txt':
            np.savetxt("pase_" + str(self.problem.kappa) + ".txt", self.u)
        elif fmt == 'hdf5':
            if os.path.isfile("./phases.h5") == True:
                print("File exist!")
                phases = h5py.File("phases.h5", 'a')
                phases.create_dataset(str(self.problem.kappa), data=self.u)
            else:
                phases = h5py.File("phases.h5", 'w')
                phases.create_dataset(str(self.problem.kappa), data=self.u)


class Plot:
    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution

        self.u = self.solution.u
        self.du = self.solution.du
        self.t = self.solution.t
        self.T = self.solution.T
        self.dt = self.solution.dt
        self.N = self.solution.N
        #self.phi = np.arctan(np.sum(np.sin(self.u),axis=1)/np.sum(np.cos(self.u),axis=1))
        self.phi = self.func()


    def func(self):
        """En clnstruccion: Metodo que conecta las partes de self.phi
           ya que esta esta a truzos.

           Esta funcion no es del todo eficiente
        """
        phi = np.arctan(np.sum(np.sin(self.u), axis=1) / np.sum(np.cos(self.u), axis=1))
        aux = copy.copy(phi)

        dphi = np.sign(phi[1:] - phi[:-1])
        s = []
        for i in xrange(len(dphi) - 1):
            if (dphi[i - 1] != dphi[i]) and (dphi[i] != dphi[i + 1]) and (np.abs(phi[i + 1] - phi[i]) > 0.5):
                s.append(i)
        s.append(len(phi))

        for i in xrange(len(s) - 1):
            diff = round((aux[s[i]] - aux[s[i] + 1]) / (.5 * np.pi), 0)
            if phi[s[i]] - phi[s[i] - 1] < 0:
                aux[s[i] + 1:s[i + 1] + 1] += diff * (np.pi/2)
            else:
                aux[s[i] + 1:s[i + 1] + 1] += diff * (np.pi/2)

        return aux


    def phase(self):
        plt.close()
        plt.figure(figsize=(7.5, 5))
        plt.plot(self.t, self.u)
        plt.xlim(0, self.T)
        plt.title("Oscillators Phase")
        plt.xlabel("Time")
        plt.ylabel(r"$\theta_1,\theta_2$")
        plt.legend([r"$\theta_1$", r"$\theta_2$"])

    def phase_mean(self):
        plt.close()
        plt.figure(figsize=(7.5, 5))
        plt.plot(self.t, self.phi)
        plt.xlim(0, self.T,'-*')
        plt.title("Fase Average")
        plt.xlabel("Time")
        plt.ylabel(r"$\psi$")

    def velocity(self):
        plt.close()
        plt.figure(figsize=(7.5, 5))
        plt.plot(self.t, self.du)
        plt.plot(self.t, np.gradient(self.phi, self.dt), '-k')
        plt.xlim(0, self.T)
        plt.ylim(np.min(self.du), np.max(self.du))
        plt.title("Aparent Angular Velocity")
        plt.xlabel("Time")
        plt.ylabel(r"$\Omega$")
        plt.legend([r"$\dot{\theta_1}$", r"$\dot{\theta_2}$", r"$\dot{\varphi}$"])

    def show(self):
        plt.show()

# class Scam_K: