import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

mu_max = 0.25
cX_max = 45.0 # g/L

a = 1.0
b = 1.0

K_s = 1.0
K_O = 17.0 * 0.08
c_O_sat = 17.0
kLa = 410.0
k_d = 0.01

y_xs = 0.5
y_ox = 131.2
m_s = 0.04

c_F = 70
c_X0 = 0.3
c_S0 = 50.0

V_0 = 2.5 #L
X_0 = c_X0 * V_0
S_0 = c_S0 * V_0
c_O_0 = 15 #mgO2/L
F_0 = 0.0
R_0 = 0.0
d_0 = 0.0

constants = {}

# Event format (start time, duration, "type", value)
event_list = [(18, 1.5, "rR", 1),(19.5,1.5,"rF",1)]

def model(y, t, constants, events):
    X, S, V, c_O, F, R, rd = y

    # Initialize event rates
    rR = 0
    rF = 0

    # Check and apply events
    for event in events:
        start_time, duration, event_type, value = event
        if start_time <= t < start_time + duration:
            if event_type == "rR":
                rR = value
            elif event_type == "rF":
                rF = value

    rV = rF - rR
    c_X = X / V

    mu = mu_max * ((S / V) / ((S / V) + K_s)) * (c_O / (c_O + K_O)) * (1 - (X / (cX_max * V)))
    rd = k_d * X
    rX = mu * X - X/V * rR - rd

    if S > 0:
        rS = -mu * (X) * (1 / y_xs + m_s / mu) - S / V * rR + c_F * rF
    else:
        S = 0
        rS = 0

    rc_O = kLa * (c_O_sat - c_O) - y_ox * X

    return [rX, rS, rV, rc_O, rF, rR, rd]

y0 = [X_0, S_0, V_0, c_O_0, F_0, R_0, d_0]

# Time points
t = np.linspace(0, 72, 1000)

# Solve ODE
solution = odeint(model, y0, t, args=(constants, event_list))

X, S, V, c_O, F, R, d = solution.T

growthrate = mu_max * ((S / V) / ((S / V) + K_s)) * (c_O / (c_O + K_O)) * (1 - (X / (cX_max * V)))

rc_O = kLa * (c_O_sat - c_O) - y_ox * growthrate

# Plotting
fig, axs = plt.subplots(5, 2, figsize=(15, 15))

# Biomass concentration (X/V)
axs[0, 0].plot(t, X/V, label='Biomass Concentration (g/L)')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Concentration (g/L)')
axs[0, 0].set_title('Biomass Concentration over Time')
axs[0, 0].grid(True)

# Substrate concentration (S/V)
axs[0, 1].plot(t, S/V, label='Substrate Concentration (g/L)')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Concentration (g/L)')
axs[0, 1].set_title('Substrate Concentration over Time')
axs[0, 1].grid(True)

# Biomass total (X)
axs[1, 0].plot(t, X, label='Biomass Total (g)')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Biomass (g)')
axs[1, 0].set_title('Biomass Total over Time')
axs[1, 0].grid(True)

# Substrate total (S)
axs[1, 1].plot(t, S, label='Substrate Total (g)')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Substrate (g)')
axs[1, 1].set_title('Substrate Total over Time')
axs[1, 1].grid(True)

# Oxygen (c_O)
axs[2, 0].plot(t, c_O, label='Oxygen (mg/L)', color='green')
axs[2, 0].set_xlabel('Time')
axs[2, 0].set_ylabel('Oxygen (mg/L)')
axs[2, 0].set_title('Oxygen Concentration over Time')
axs[2, 0].grid(True)

# Growthrate (mu)
axs[2, 1].plot(t, growthrate, label='Feed Totalizer (L)', color='orange')
axs[2, 1].set_xlabel('Time')
axs[2, 1].set_ylabel('Growth rate (mu)')
axs[2, 1].set_title('Growth rate over Time')
axs[2, 1].grid(True)

# Feed totalizer (F)
axs[3, 1].plot(t, F, label='Feed Totalizer (L)', color='orange')
axs[3, 1].set_xlabel('Time')
axs[3, 1].set_ylabel('Feed (L)')
axs[3, 1].set_title('Feed Totalizer over Time')
axs[3, 1].grid(True)

# Removal totalizer (R)
axs[3, 0].plot(t, R, label='Removal Totalizer (L)', color='red')
axs[3, 0].set_xlabel('Time')
axs[3, 0].set_ylabel('Removal (L)')
axs[3, 0].set_title('Removal Totalizer over Time')
axs[3, 0].grid(True)

# Removal totalizer (R)
axs[4, 0].plot(t, V, label='Removal Totalizer (L)', color='red')
axs[4, 0].set_xlabel('Time')
axs[4, 0].set_ylabel('Total Volume (L)')
axs[4, 0].set_title('Total Volume over Time')
axs[4, 0].grid(True)

# Adjust layout
plt.tight_layout()

plt.show()
