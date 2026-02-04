# this generates ou process that was discretized via euler maruyama. 
# i reckon i will be using velocity magnitude? to get stress magnitude so i would like X_t to remain positive, hence mean is significantly larger than sd.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.interpolate import griddata

theta = 1
mu = 5
sigma = 0.3
X0 = 5
Y0 = 5
T = 10
dt = 0.01
N = int(T/dt)

X = np.zeros(N)
Y = np.zeros(N)
X[0] = X0
Y[0] = Y0

np.random.seed(1)

for t in range(1,N):
    Wt = np.random.normal(0,1)
    X[t] = X[t-1] + theta * (mu -X[t-1]) * dt + sigma * np.sqrt(dt) * Wt 
    Y[t] = Y[t-1] + X[t-1] * dt

plt.figure()
plt.plot(np.linspace(0,T,N),X, label = "ou process")
plt.plot(np.linspace(0,T,N),Y, label = "integrated ou process")
plt.xlabel("t")
#plt.ylabel("X")
plt.title("OU process")
plt.legend()
plt.show()

# i can compute stress on my biofilm so i can choose a single point and i have stress for shear velocity 1500 (i also have 0 stress for 0 velocity)
# now lets say i have linear relationship stress = m*velocity + b 
# i need to take my X[t] and put this into my linear relationship

# df = pd.read_csv("steady_stokes_results/uneven_boundary_stress.csv")
# df["m"] = df["stress_mag"] / 1500
# df["b"] = 0.0

# m = df["m"].iloc[0]
# b = 0

# def linear_relationship(x):
#     stress = m * x + b
#     return stress

# s = np.zeros(N)

# for i in range(N):
#     s[i] = linear_relationship(X[i])

# plt.plot(np.linspace(0,T,N), s)
# plt.xlabel("t")
# plt.ylabel("s")
# plt.title("Stress Time Series")
# plt.show()

# # now if i have multiple points, i have multiple linear relationships. i can create some sort of 2d map?

# df = pd.read_csv("steady_stokes_results/uneven_boundary_stress.csv")
# df["m"] = df["stress_mag"] / 1500
# df["b"] = 0.0

# m = df["m"].to_numpy()
# Np = len(m)

# stress_time = np.outer(m,X)


# plt.imshow(
#     stress_time.T,
#     aspect="auto",
#     origin="lower",
#     extent=[0, T, 0, Np],
#     cmap="viridis"
# )
# plt.colorbar(label="stress")
# plt.xlabel("time")
# plt.ylabel("biofilm point index")
# plt.title("Stress fluctuations on biofilm")
# plt.show()

# x = df["x"].to_numpy()
# y = df["y"].to_numpy()
# grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), 100),
#                              np.linspace(y.min(), y.max(), 50)) 

# stress_at_t = stress_time[:, -1]

# grid_stress = griddata((x, y), stress_at_t, (grid_x, grid_y), method='cubic')

# plt.figure(figsize=(6,4))
# plt.pcolormesh(grid_x, grid_y, grid_stress, shading='auto', cmap='viridis')
# plt.colorbar(label='Stress')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'Biofilm stress at t={T:.2f}s')
# plt.show()

# this bit is just testing using a pdf instead of a generated ou process. the pdf will be gaussian and have same mu and sigma as ou process.

# mu_pdf = 5.0      # mean velocity
# sigma_pdf = 0.3   # std of velocity
# N_pdf = 1000

# np.random.seed(1)
# velocity_samples = np.random.normal(mu, sigma, size=N)

# plt.plot(velocity_samples)
# plt.xlabel("Time step")
# plt.ylabel("Velocity (m/s)")
# plt.title("Random velocity time series sampled from Gaussian")
# plt.show()

# plt.figure(figsize=(10,4))
# t_vals = np.linspace(0, T, N)

# plt.plot(t_vals, X, label="OU process (correlated)")
# plt.plot(t_vals, velocity_samples, label="Gaussian samples (uncorrelated)", alpha=0.7)
# plt.xlabel("t")
# plt.ylabel("X")
# plt.title("Comparison: OU process vs Gaussian sampling")
# plt.legend()
# plt.show()

# print("OU mean:", np.mean(X))
# print("OU std :", np.std(X))

# print("Gaussian mean:", np.mean(velocity_samples))
# print("Gaussian std :", np.std(velocity_samples))