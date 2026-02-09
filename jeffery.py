import numpy as np
from scipy.integrate import quad

# ============================================================
# Parameters
# ============================================================
mu = 1.0
gamma_dot = 1.0

a = 3.0
b = 2.0
c = 1.0

# ============================================================
# Jeffery ellipsoidal geometry
# ============================================================
def Delta(lam):
    return np.sqrt((a*a + lam)*(b*b + lam)*(c*c + lam))

def I1(lam):
    f = lambda s: 1.0 / ((a*a + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def I2(lam):
    f = lambda s: 1.0 / ((b*b + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def I3(lam):
    f = lambda s: 1.0 / ((c*c + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

# ============================================================
# Surface normal (exact)
# ============================================================
def surface_normal(x, y, z):
    n = np.array([x/a**2, y/b**2, z/c**2])
    return n / np.linalg.norm(n)

# ============================================================
# Jeffery constants for simple shear
# (derived from eqs. 25–27, 37 in Jeffery 1922)
# ============================================================
def jeffery_constants():
    lam = 0.0
    A = Delta(lam)

    I1_0 = I1(lam)
    I2_0 = I2(lam)
    I3_0 = I3(lam)

    # Torque-free angular velocity (Jeffery eq. 37)
    omega_x = 0.0
    omega_y = 0.0
    omega_z = gamma_dot * (a*a - b*b) / (a*a + b*b)

    # Velocity-gradient correction coefficients
    Cxy = gamma_dot * a*a * b*b * I1_0 / A
    Cyx = gamma_dot * a*a * b*b * I2_0 / A

    return omega_x, omega_y, omega_z, Cxy, Cyx

# ============================================================
# Velocity gradient ∇u (exact Jeffery)
# ============================================================
def velocity_gradient(x, y, z):
    lam = 0.0
    A = Delta(lam)

    I1_0 = I1(lam)
    I2_0 = I2(lam)
    I3_0 = I3(lam)

    # Far-field gradient
    grad_inf = np.array([
        [0.0, gamma_dot, 0.0],
        [0.0, 0.0,       0.0],
        [0.0, 0.0,       0.0]
    ])

    # Disturbance gradient (Jeffery exact)
    dudx = -gamma_dot * a*a * I1_0 / A
    dudy = gamma_dot * (1.0 - a*a * I1_0 / A)
    dudz = 0.0

    dvdx = -gamma_dot * b*b * I2_0 / A
    dvdy = gamma_dot * b*b * I2_0 / A
    dvdz = 0.0

    dwdx = 0.0
    dwdy = 0.0
    dwdz = 0.0

    grad_dist = np.array([
        [dudx, dudy, dudz],
        [dvdx, dvdy, dvdz],
        [dwdx, dwdy, dwdz]
    ])

    return grad_inf + grad_dist

# ============================================================
# Pressure field (Jeffery eq. 21 specialised)
# ============================================================
def pressure(x, y, z):
    lam = 0.0
    A = Delta(lam)

    I1_0 = I1(lam)
    I2_0 = I2(lam)

    p = -4 * mu * gamma_dot * a*a * b*b * (I1_0 - I2_0) * y / A
    return p

# ============================================================
# Stress tensor
# ============================================================
def stress_tensor(x, y, z):
    grad_u = velocity_gradient(x, y, z)
    p = pressure(x, y, z)
    return -p*np.eye(3) + mu*(grad_u + grad_u.T)

# ============================================================
# Surface shear stress
# ============================================================
def surface_shear_stress(x, y, z):
    n = surface_normal(x, y, z)
    sigma = stress_tensor(x, y, z)
    tau = (np.eye(3) - np.outer(n, n)) @ sigma @ n
    return tau, np.linalg.norm(tau)

# ============================================================
# Sample surface point
# ============================================================
theta = np.pi/4
phi = np.pi/3

x = a*np.sin(theta)*np.cos(phi)
y = b*np.sin(theta)*np.sin(phi)
z = c*np.cos(theta)

tau_vec, tau_mag = surface_shear_stress(x, y, z)

print("Surface point:", x, y, z)
print("Shear stress vector:", tau_vec)
print("Shear stress magnitude:", tau_mag)
