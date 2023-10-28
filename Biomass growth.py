import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math



def logistic_growth(k,x_o,x_max,t_frame, increment):
    
    t = np.arange(0,t_frame,increment)

    x_array = []
    for i in t:
        x = (x_o*math.exp(k*i))/(1-(x_o/x_max)*(1-math.exp(k*i)))
        x_array.append(x)
    
    return x_array, t

def exponential_growth(k,x_o,t_frame,increment):
    
    t = np.arange(0,t_frame,increment)

    x_array = []

    for i in t:
        x = x_o*math.exp(k*i)
        x_array.append(x)

    return x_array, t



def cons_model(xt,y_xs,concentration,ms):

    consumed = [0]
    biomass = xt[0]
    cons_tot = [0]
    
    i=0
    for x_t in biomass:
        try:
            x_dt = biomass[i+1]-biomass[i]
            cons = x_dt/y_xs/concentration + biomass[i]*ms
            consumed.append(cons)
            cons_tot.append(cons_tot[i] + cons)

            i = i+1
        except:
            pass
    return consumed, cons_tot



x, t = logistic_growth(0.2,0.3*4.1,30*4.1,40,1)

# x, t = exponential_growth(0.2,0.3*45,24,1)

plt.plot(t,x)

s_dt, s = cons_model([x,t],0.5,0.53,0.04)

plt.plot(t, s)
plt.show()    