# this generates ou process that was discretized via euler maruyama. 
# i reckon i will be using velocity magnitude? to get stress magnitude so i would like X_t to remain positive, hence mean is significantly larger than sd.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

theta = 1
mu = 5
sigma = 0.3
X0 = 5
T = 10
dt = 0.01
N = int(T/dt)

X = np.zeros(N)
X[0] = X0

for t in range(1,N):
    Wt = np.random.normal(0,1)
    X[t] = X[t-1] + theta * (mu -X[t-1]) * dt + sigma * np.sqrt(dt) * Wt 

plt.plot(np.linspace(0,T,N),X)
plt.xlabel("t")
plt.ylabel("X")
plt.show()

# i can compute stress on my biofilm so i can choose a single point and i have stress for shear velocity 1500 (i also have 0 stress for 0 velocity)
# now lets say i have linear relationship stress = m*velocity + b 
# i need to take my X[t] and put this into my linear relationship

m = 1
b = 1 

def linear_relationship(x):
    stress = m * x + b
    return stress

s = np.zeros(N)

for i in range(N):
    s[i] = linear_relationship(X[i])

plt.plot(np.linspace(0,T,N), s)
plt.xlabel("t")
plt.ylabel("s")
plt.show()

# now if i have multiple points, i have multiple linear relationships. i can create some sort of 2d map?

df = pd.read_csv("steady_stokes_results/elliptic_boundary_stress.csv")
df["m"] = df["stress_mag"] / 1500
df["b"] = 0.0

m = df["m"].to_numpy()
Np = len(m)

stress_time = np.outer(m,X)


plt.imshow(
    stress_time.T,
    aspect="auto",
    origin="lower",
    extent=[0, T, 0, Np],
    cmap="viridis"
)
plt.colorbar(label="stress")
plt.xlabel("time")
plt.ylabel("biofilm point index")
plt.title("Stress fluctuations on biofilm")
plt.show()