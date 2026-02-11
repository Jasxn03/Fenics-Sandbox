import numpy as np
from scipy.integrate import quad
import sympy as sp
import matplotlib.pyplot as plt


mu = 1.0
gamma_dot = 1.0

a = 3.0
b = 2.0
c = 1.0

# geometrical integrals
def Delta(lam):
    return np.sqrt((a*a + lam)*(b*b + lam)*(c*c + lam))

def alpha(lam):
    f = lambda s: 1.0 / ((a*a + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def beta(lam):
    f = lambda s: 1.0 / ((b*b + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def gamma(lam):
    f = lambda s: 1.0 / ((c*c + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def alpha_prime(lam):
    f = lambda s: 1.0 / ((b*b + s)*(c*c+s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def beta_prime(lam):
    f = lambda s: 1.0 / ((a*a + s)*(c*c +s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def gamma_prime(lam):
    f = lambda s: 1.0 / ((a*a + s)*(b*b + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def alpha_prime2(lam):
    f = lambda s: s / ((b*b + s)*(c*c +s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def beta_prime2(lam):
    f = lambda s: s / ((a*a + s)*(c*c +s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def gamma_prime2(lam):
    f = lambda s: s / ((a*a + s)*(b*b + s)*Delta(s))
    return quad(f, lam, np.inf, limit=200)[0]

def P_squared(x,y,z,lam):
    return 1/(((x*x)/(a*a + lam)**2) + ((y*y)/(b*b + lam)**2) + ((z*z)/(c*c + lam)**2))

# axis m_i and n_i (assumed that ellipsoid is aligned with the flow which makes sense for me?)
m_1 = 1
n_1 = 0
m_2 = 0
n_2 = 1
m_3 = 0
n_3 = 0

# shear flow velocity
kappa = 1500

# flow field coefficients
a_bold = kappa * m_1 * n_1
b_bold = kappa * m_2 * n_2
c_bold = kappa * m_3 * n_3
f_bold = 1/2 * kappa * (m_2 * n_3 + m_3 * n_2)
g_bold = 1/2 * kappa * (m_3 * n_1 + m_1 * n_3)
h_bold = 1/2 * kappa * (m_1 * n_2 + m_2 * n_1)
xii = 1/2 * kappa * (m_2 * n_3 - m_3 * n_2)
eta = 1/2 * kappa * (m_3 * n_1 - m_1 * n_3)
xi = 1/2 * kappa * (m_1 * n_2 - m_2 * n_1)


# Surface normal 

def surface_normal(x, y, z):
    n = np.array([x/a**2, y/b**2, z/c**2])
    return n / np.linalg.norm(n)


# Jeffery constants for simple shear

A = 1/6 * (2*alpha_prime2(0)* a_bold - beta_prime2(0)*b_bold - gamma_prime2(0)*c_bold)/(beta_prime2(0)*gamma_prime2(0) + gamma_prime2(0)*alpha_prime2(0)+alpha_prime2(0)*beta_prime2(0))
B = 1/6 * (2*beta_prime2(0)* b_bold - alpha_prime2(0)*a_bold - gamma_prime2(0)*c_bold)/(beta_prime2(0)*gamma_prime2(0) + gamma_prime2(0)*alpha_prime2(0)+alpha_prime2(0)*beta_prime2(0))
C = 1/6 * (2*gamma_prime2(0)* c_bold - beta_prime2(0)*b_bold - alpha_prime2(0)*a_bold)/(beta_prime2(0)*gamma_prime2(0) + gamma_prime2(0)*alpha_prime2(0)+alpha_prime2(0)*beta_prime2(0))
F = f_bold/(2*alpha_prime(0)*(b*b + c*c))
G = g_bold/(2*beta_prime(0)*(a*a + c*c))
H = h_bold/(2*gamma_prime(0)*(b*b + a*a))
R = -f_bold/alpha_prime(0)
S = -g_bold/beta_prime(0)
T = -h_bold/gamma_prime(0)
U = 2*b*b*B - 2*c*c*C
V = 2*c*c*C - 2*a*a*A
W = 2*a*a*A - 2*b*b*B


# flow field on ellipsoid boundary (so lambda = 0)
def flow_field(x,y,z):
    alpha0 = alpha(0)
    alpha_prime0 = alpha_prime(0)
    beta0 = beta(0)
    beta_prime0 = beta_prime(0)
    gamma0 = gamma(0)
    gamma_prime0 = gamma_prime(0)
    u =( x*(a_bold + gamma_prime0*W - beta_prime0*V - 2*(alpha0+beta0+gamma0)*A) 
            + y*(h_bold - xi + gamma_prime0*T - 2*beta0*H + 2*alpha0*H) 
            + z*(g_bold + eta + beta_prime0*S - 2*gamma0 * G + 2*alpha0 * G)
            - (2*x*P_squared(x,y,z,0))/(a*a * Delta(0)) * ((R + 2*b*b*F + 2*c*c*F) * y*z/(b*b*c*c)
                                            + (S + 2*c*c*G + 2*a*a*G) * z*x/(c*c*a*a)
                                            + (T + 2*a*a*H + 2*b*b*H) * x*y/(a*a*b*b)
                                            + (W- 2*a*a*A + 2*b*b*B) * y*y/(b*b*b*b)
                                            - (V - 2*c*c*C + 2*a*a*A) * z*z/(c*c*c*c)

            )
            )
    v =( x*(h_bold + xi + gamma_prime0*T + 2*beta0*H - 2*alpha0*H) 
            + y*(b_bold - alpha_prime0*U - gamma_prime0*W - 2*(alpha0 + beta0 + gamma0)*B) 
            + z*(f_bold + xii + alpha_prime0*R - 2*gamma0 * F + 2*beta0 * F)
            - (2*y*P_squared(x,y,z,0))/(b*b * Delta(0)) * ((R + 2*b*b*F + 2*c*c*F) * y*z/(b*b*c*c)
                                            + (S + 2*c*c*G + 2*a*a*G) * z*x/(c*c*a*a)
                                            + (T + 2*a*a*H + 2*b*b*H) * x*y/(a*a*b*b)
                                            + (U- 2*b*b*B + 2*c*c*C) * z*z/(c*c*c*c)
                                            - (W - 2*a*a*A + 2*b*b*B) * x*x/(a*a*a*a)

            )
            )
    w =( x*(g_bold - eta + beta_prime0*S - 2*alpha0*G + 2*gamma0 * G) 
            + y*(f_bold + xii + alpha_prime0*R + 2*gamma0*F - 2*beta0*F) 
            + z*(c_bold + beta_prime0*V - alpha_prime0*U - 2*(alpha0 + beta0 + gamma0)*C)
            - (2*z*P_squared(x,y,z,0))/(c*c * Delta(0)) * ((R + 2*b*b*F + 2*c*c*F) * y*z/(b*b*c*c)
                                            + (S + 2*c*c*G + 2*a*a*G) * z*x/(c*c*a*a)
                                            + (T + 2*a*a*H + 2*b*b*H) * x*y/(a*a*b*b)
                                            + (V - 2*c*c*C + 2*a*a*A) * x*x/(a*a*a*a)
                                            - (U - 2*b*b*B + 2*c*c*C) * y*y/(b*b*b*b)

            )
            )
    return u ,v ,w


# Velocity gradient (finite difference approx)

def velocity_gradient(x, y, z, h=1e-6):
    u0, v0, w0 = flow_field(x, y, z)

    ux = (flow_field(x+h, y, z)[0] - u0) / h
    uy = (flow_field(x, y+h, z)[0] - u0) / h
    uz = (flow_field(x, y, z+h)[0] - u0) / h

    vx = (flow_field(x+h, y, z)[1] - v0) / h
    vy = (flow_field(x, y+h, z)[1] - v0) / h
    vz = (flow_field(x, y, z+h)[1] - v0) / h

    wx = (flow_field(x+h, y, z)[2] - w0) / h
    wy = (flow_field(x, y+h, z)[2] - w0) / h
    wz = (flow_field(x, y, z+h)[2] - w0) / h

    return np.array([
        [ux, uy, uz],
        [vx, vy, vz],
        [wx, wy, wz]
    ])


# Stress

def stress_tensor(x, y, z):
    grad_u = velocity_gradient(x, y, z)
    return mu*(grad_u + grad_u.T)

def traction(x, y, z):
    n = surface_normal(x, y, z)
    sigma_v = stress_tensor(x, y, z)
    return sigma_v @ n

def surface_shear_stress(x, y, z):
    n = surface_normal(x, y, z)
    t = traction(x, y, z)
    normal_comp = np.dot(t,n) *n
    shear_comp = t -normal_comp
    return np.linalg.norm(shear_comp)

# Sample surface point

theta = np.pi/4
phi = np.pi/3

x = a*np.sin(theta)*np.cos(phi)
y = b*np.sin(theta)*np.sin(phi)
z = c*np.cos(theta)

stress_mag = surface_shear_stress(x, y, z)
u, v, w = flow_field(x,y,z)
velocity_mag = np.sqrt(u*u + v*v + w*w)

print("Surface point:", x, y, z)
grad = velocity_gradient(x, y, z)
print("Velocity gradient:", grad)

print("Shear stress magnitude:", stress_mag)
print("Velocity magnitude:", velocity_mag)


nx, ny = 50, 50
x_vals = np.linspace(-a, a, nx)
y_vals = np.linspace(-b, b, ny)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X) 

velocity_mag_grid = np.zeros_like(X)
shear_stress_grid = np.zeros_like(X)

for i in range(nx):
    for j in range(ny):
        x, y, z = X[j,i], Y[j,i], Z[j,i]
        if (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1.0:
            u, v, w = flow_field(x, y, z)
            velocity_mag_grid[j,i] = np.sqrt(u**2 + v**2 + w**2)
            shear_stress_grid[j,i] = surface_shear_stress(x, y, z)
        else:
            velocity_mag_grid[j,i] = np.nan
            shear_stress_grid[j,i] = np.nan

plt.figure(figsize=(6,5))
plt.contourf(X, Y, shear_stress_grid, levels=50, cmap='inferno')
plt.colorbar(label='Shear stress magnitude')
plt.title('Surface Shear Stress on x-y Plane (z=0)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6,5))
plt.contourf(X, Y, velocity_mag_grid, levels=50, cmap='inferno')
plt.colorbar(label='Velocity magnitude')
plt.title('Velocity Magnitude on x-y Plane (z=0)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
