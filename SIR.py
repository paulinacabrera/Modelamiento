# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 05:37:42 2019

@author: Paulina
"""

import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma, N):
    S, I, R = y 
    dS_dt = -beta*(S*I/N)
    dI_dt = beta*(S*I/N) - gamma*I
    dR_dt = gamma*I
    
    
    return([dS_dt, dI_dt, dR_dt])
    
#Condiciones Iniciales
    
S0 = np.array([12099.996, 6299.994, 4499.997])
I0 = np.array([4, 6, 3])
R0 = np.array([0,0,0])
beta =  np.array([0.27, 0.45, 0.28])
gamma = 0.18
N = np.array([12100, 6300, 4500])

#Tiempo

t = np.linspace(0, 200, 1000)

#Resultado

solution_g = scipy.integrate.odeint(SIR_model, [S0[0], I0[0], R0[0]], t, args = (beta[0], gamma, N[0]))


solution_s = scipy.integrate.odeint(SIR_model, [S0[1], I0[1], R0[1]], t, args = (beta[1], gamma, N[1]))


solution_l = scipy.integrate.odeint(SIR_model, [S0[2], I0[2], R0[2]], t, args = (beta[2], gamma, N[2]))

solution = np.array([solution_g, solution_s, solution_l])

# gráficos
w = 7
h = 6
d = 80

fig = plt.figure(figsize=(w, h), dpi=d)

ax1 = fig.add_subplot(3,1,1)
ax1.set_ylim(0,12100)
plt.plot(t, solution[0][:,0], label = "S(t)")
plt.plot(t, solution[0][:,1], label = "I(t)")
plt.plot(t, solution[0][:,2], label = "R(t)")
ax1.legend(loc='right')
ax1.axes.set_xticklabels([],[])
ax1.set_title("Guinea")

ax2 = fig.add_subplot(3,1,2)
ax2.set_ylim(0,6300)
plt.plot(t, solution[1][:,0], label = "S(t)")
plt.plot(t, solution[1][:,1], label = "I(t)")
plt.plot(t, solution[1][:,2], label = "R(t)")
ax2.axes.set_xticklabels([],[])
ax2.legend(loc='right')
ax2.axes.set_ylabel('Población (miles)')
ax2.set_title("Sierra Leon")

ax3 = fig.add_subplot(3,1,3)
ax3.set_ylim(0,4500)
plt.plot(t, solution[2][:,0], label = "S(t)")
plt.plot(t, solution[2][:,1], label = "I(t)")
plt.plot(t, solution[2][:,2], label = "R(t)")
ax3.legend(loc='right')
ax3.set_title("Liberia")
ax3.axes.set_xlabel('Tiempo (días)')
plt.savefig('sirmodeloevd.jpg')




