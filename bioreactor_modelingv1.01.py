import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint





### initial parameters


cx_0 = 0.31 # g/L

cfs_0 = 0.53 # g Gluc /g F

rho_f = 1.22 # g/mL # kg F/L F

V_0 = 450 # L
F_0 = 20000 # g F 
X_0 = cx_0*V_0 # g
cs_0 = F_0*cfs_0/V_0 # g Gluc / L tot
S_0 = cs_0*V_0 # g Gluc

cf_0 = F_0/V_0 # g F/L
cO_0 = 17 # mg/L
c_DeX_0 = 0



### constants
y_xs = 0.53 # gX/gS
c_Osat = 17 # mgO2/L
y_xo = 1/131 # gX/mgO2
m_s = 0.04 # gS/gX/h
k_s = 0.2 # g/L
k_b = 0.5
k_O = 0.8 #mgO2/L
kLa = 180 # 1/h
k_f = 0.2
X_max = 45 # g/L
mu_max = 0.25
d_c = 0.04


def model(y,t, mu_max, k_s, X_max, m_s, y_xs,):
    S,c_s, X,c_x,F,V, c_O,c_DeX = y
    # O, V, F, R


    ## define a feed rate that changes at a certain time
    if t>17:
        dFdt = k_s * F #gSol/ h
    if t>20:
        dFdt = 300 #gSol/ h
    else:
        dFdt = 0
        # dFdt = k_f * F # g/L/h
    

    dVdt = dFdt/rho_f/1000

    mu = mu_max * (c_s / (c_s+k_s)) * (1-c_x/X_max) * (c_O / (c_O+k_O))

    dc_DeXdt = d_c * c_x
    # g/L/h
    dc_xdt = mu * c_x - dVdt*c_x/V
    dXdt = dc_xdt*V 



    dc_sdt = -dc_xdt/y_xs - m_s*c_x + dFdt/cfs_0/V
    dSdt = dc_sdt*V
    
    dc_Odt = kLa*(c_Osat-c_O) - c_x/y_xo
    # if c_O > c_Osat*0.2:


    # dVdt = None

    # dFdt = 0c
    
    # dRdt = 0
    return [dSdt, dc_sdt, dXdt, dc_xdt, dFdt,dVdt, dc_Odt,dc_DeXdt]


y0 = [S_0, cs_0,X_0,cx_0, F_0, V_0, cO_0,c_DeX_0]
# [S_0, X_0, V_0, F_0, O_0, R_0, D_0]

# Time points
t = np.linspace(0, 30, 1000)

# Solve ODE
solution = odeint(model, y0, t, args=(mu_max, k_s, X_max, m_s, y_xs))

S, c_s, X, c_x ,F, V, c_O , c_DeX= solution.T

# Calculate additional quantities for plotting

X_real = [(0, 0.32),(16,11.82),(18,14.2),(21,19.67),(22,20.53)]
sample_tps, samples = zip(*X_real)

# Plot the results
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Substrate concentration (S)
axs[0, 0].plot(t, c_s, label='Substrate (S)')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Concentration (g/L)')
axs[0, 0].set_title('Substrate (S) over Time')
axs[0, 0].grid(True)

axs[0, 1].plot(t, c_x, label='Biomass (X)')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Concentration (g/L)')
axs[0, 1].set_title('Biomass (g/L) over Time')
axs[0, 1].grid(True)

axs[0, 1].scatter(sample_tps,samples, label="Real biomass (g/L)")
axs[0, 1].grid(True)

axs[1, 1].plot(t, F, label='Feed (F)')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Total fed(g)')
axs[1, 1].set_title('Feed (g) over Time')
axs[1, 1].grid(True)

axs[1, 0].plot(t, V, label='Volume (V)')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Volume (L)')
axs[1, 0].set_title('Volume over time (V)')
axs[1, 0].grid(True)

# axs[2, 1].plot(t, F, label='Feed (F)')
# axs[2, 1].set_xlabel('Time')
# axs[2, 1].set_ylabel('Total fed(g)')
# axs[2, 1].set_title('Feed (g) over Time')
# axs[2, 1].grid(True)

axs[2, 0].plot(t, c_O, label='Oxygen')
axs[2, 0].set_xlabel('Time')
axs[2, 0].set_ylabel('Oxygen (mg/L)')
axs[2, 0].set_title('Oxygen over time (mg/L)')
axs[2, 0].grid(True)


plt.tight_layout()
plt.show()