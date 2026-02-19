import numpy as np
from scipy.integrate import quad
import sympy as sp
import matplotlib.pyplot as plt
import time
import csv

mu = 1.0
a = 1.0
b = 1.0
c = 1.0

#region Coefficient definitions

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

# Jeffery constants for simple shear

alpha0 = alpha(0)
beta0 = beta(0)
gamma0 = gamma(0)
alpha_prime0 = alpha_prime(0)
beta_prime0 = beta_prime(0)
gamma_prime0 = gamma_prime(0)
alpha_prime2_0 = alpha_prime2(0)
beta_prime2_0 = beta_prime2(0)
gamma_prime2_0 = gamma_prime2(0)


A = 1/6 * (2*alpha_prime2_0* a_bold - beta_prime2_0*b_bold - gamma_prime2_0*c_bold)/(beta_prime2_0*gamma_prime2_0 + gamma_prime2_0*alpha_prime2_0+alpha_prime2_0*beta_prime2_0)
B = 1/6 * (2*beta_prime2_0* b_bold - alpha_prime2_0*a_bold - gamma_prime2_0*c_bold)/(beta_prime2_0*gamma_prime2_0 + gamma_prime2_0*alpha_prime2_0+alpha_prime2_0*beta_prime2_0)
C = 1/6 * (2*gamma_prime2_0* c_bold - beta_prime2_0*b_bold - alpha_prime2_0*a_bold)/(beta_prime2_0*gamma_prime2_0 + gamma_prime2_0*alpha_prime2_0+alpha_prime2_0*beta_prime2_0)
F = f_bold/(2*alpha_prime0*(b*b + c*c))
G = g_bold/(2*beta_prime0*(a*a + c*c))
H = h_bold/(2*gamma_prime0*(b*b + a*a))
R = -f_bold/alpha_prime0
S = -g_bold/beta_prime0
T = -h_bold/gamma_prime0
U = 2*b*b*B - 2*c*c*C
V = 2*c*c*C - 2*a*a*A
W = 2*a*a*A - 2*b*b*B

#endregion

#region Flow field function

# flow field 
def flow_field(x,y,z, lam):
    u =( x*(a_bold + gamma_prime(lam)*W - beta_prime(lam)*V - 2*(alpha(lam)+beta(lam)+gamma(lam))*A) 
            + y*(h_bold - xi + gamma_prime(lam)*T - 2*beta(lam)*H + 2*alpha(lam)*H) 
            + z*(g_bold + eta + beta_prime(lam)*S - 2*gamma(lam) * G + 2*alpha(lam) * G)
            - (2*x*P_squared(x,y,z,lam))/((a*a+lam) * Delta(lam)) * ((R + 2*(b*b+lam)*F + 2*(c*c+lam)*F) * y*z/((b*b+lam)*(c*c+lam))
                                            + (S + 2*(c*c+lam)*G + 2*(a*a+lam)*G) * z*x/((c*c+lam)*(a*a+lam))
                                            + (T + 2*(a*a + lam)*H + 2*(b*b + lam)*H) * x*y/((a*a+lam)*(b*b+lam))
                                            + (W- 2*(a*a+lam)*A + 2*(b*b+lam)*B) * y*y/((b*b+lam)*(b*b+lam))
                                            - (V - 2*(c*c+lam)*C + 2*(a*a+lam)*A) * z*z/((c*c+lam)*(c*c+lam))

            )
            )
    v =( x*(h_bold + xi + gamma_prime(lam)*T + 2*beta(lam)*H - 2*alpha(lam)*H) 
            + y*(b_bold - alpha_prime(lam)*U - gamma_prime(lam)*W - 2*(alpha(lam) + beta(lam) + gamma(lam))*B) 
            + z*(f_bold + xii + alpha_prime(lam)*R - 2*gamma(lam) * F + 2*beta(lam) * F)
            - (2*y*P_squared(x,y,z,lam))/(b*b * Delta(lam)) * ((R + 2*(b*b+lam)*F + 2*(c*c+lam)*F) * y*z/((b*b+lam)*(c*c+lam))
                                            + (S + 2*(c*c+lam)*G + 2*(a*a+lam)*G) * z*x/((c*c+lam)*(a*a+lam))
                                            + (T + 2*(a*a+lam)*H + 2*(b*b+lam)*H) * x*y/((a*a + lam)*(b*b+lam))
                                            + (U- 2*(b*b+lam)*B + 2*(c*c+lam)*C) * z*z/((c*c+lam)*(c*c+lam))
                                            - (W - 2*(a*a+lam)*A + 2*(b*b+lam)*B) * x*x/((a*a+lam)*(a*a+lam))

            )
            )
    w =( x*(g_bold - eta + beta_prime(lam)*S - 2*alpha(lam)*G + 2*gamma(lam) * G) 
            + y*(f_bold + xii + alpha_prime(lam)*R + 2*gamma(lam)*F - 2*beta(lam)*F) 
            + z*(c_bold + beta_prime(lam)*V - alpha_prime(lam)*U - 2*(alpha(lam) + beta(lam) + gamma(lam))*C)
            - (2*z*P_squared(x,y,z,lam))/(c*c * Delta(lam)) * ((R + 2*(b*b + lam)*F + 2*(c*c+lam)*F) * y*z/((b*b+lam)*(c*c+lam))
                                            + (S + 2*(c*c+lam)*G + 2*(a*a+lam)*G) * z*x/((c*c+lam)*(a*a+lam))
                                            + (T + 2*(a*a + lam)*H + 2*(b*b+lam)*H) * x*y/((a*a+lam)*(b*b+lam))
                                            + (V - 2*(c*c+lam)*C + 2*(a*a+lam)*A) * x*x/((a*a+lam)*(a*a+lam))
                                            - (U - 2*(b*b+lam)*B + 2*(c*c+lam)*C) * y*y/((b*b+lam)*(b*b+lam))

            )
            )
    return u ,v ,w

#endregion

#region Plotting

# nx, ny = 10, 10
# x_vals = np.linspace(-5, 5, nx)
# y_vals = np.linspace(-5, 5, ny)
# X, Y = np.meshgrid(x_vals, y_vals)
# Z = np.zeros_like(X) 

# velocity_mag_grid = np.zeros_like(X)
# shear_stress_grid = np.zeros_like(X)

# for i in range(nx):
#     for j in range(ny):
#         x, y, z = X[j,i], Y[j,i], Z[j,i]
#         c1 = a*a + b*b - x*x - y*y # this is fine to do as a quadratic because i am cutting z=0 plane so all z=0 so i get quadratic
#         c2 = a*a*b*b - b*b*x*x - a*a*y*y
#         lam1 = (-c1 + np.sqrt(c1*c1 - 4*c2))/2
#         lam2 = (-c1 - np.sqrt(c1*c1 - 4*c2))/2
#         lam = np.max([lam1, lam2])
#         if (x/a)**2 + (y/b)**2 + (z/c)**2 >= 1.0:
#             u, v, w = flow_field(x, y, z, lam)
#             velocity_mag_grid[j,i] = np.sqrt(u**2 + v**2 + w**2)
#             shear_stress_grid[j,i] = surface_shear_stress(x, y, z, lam)
#         else:
#             velocity_mag_grid[j,i] = np.nan
#             shear_stress_grid[j,i] = np.nan

# # plt.figure(figsize=(6,5))
# # plt.contourf(X, Y, shear_stress_grid, levels=50, cmap='viridis')
# # plt.colorbar(label='Shear stress magnitude')
# # plt.title('Surface Shear Stress on x-y Plane (z=0)')
# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.axis('equal')
# # plt.show()

# plt.figure(figsize=(6,6))
# plt.contourf(X, Y, velocity_mag_grid, levels=50, cmap='viridis')
# plt.colorbar(label='Velocity magnitude')
# plt.title('Velocity Magnitude on x-y Plane (z=0)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')
# plt.show()

# above this line was commented out
#--------------------------------------------------------------------------------------------------------

# nx, ny, nz = 100, 100, 100 # about 11 mins for 100x100x100
# x_vals = np.linspace(-5, 5, nx)
# y_vals = np.linspace(-5, 5, ny)
# z_vals = np.linspace(-5, 5, nz)
# X_xz, Z_xz = np.meshgrid(x_vals, z_vals)
# Y0 = np.zeros_like(X_xz) 
# X_xy, Y_xy = np.meshgrid(x_vals, y_vals)
# Z0 = np.zeros_like(X_xy) 
# Y_yz, Z_yz = np.meshgrid(y_vals, z_vals)
# X0 = np.zeros_like(Y_yz) 

# velocity_xy = np.zeros_like(X_xy)
# velocity_xz = np.zeros_like(X_xz)
# velocity_yz = np.zeros_like(Y_yz)

# time_now = time.ctime() 
# print(f"time: {time_now}")
# start_time = time.time()

# for i in range(nx):
#     for j in range(nz):
#         x, y, z = X_xz[j,i], Y0[j,i], Z_xz[j,i]
#         c1 = a*a + c*c - x*x - z*z 
#         c2 = a*a*c*c - a*a*z*z - c*c*x*x
#         lam1 = (-c1 + np.sqrt(c1*c1 - 4*c2))/2
#         lam2 = (-c1 - np.sqrt(c1*c1 - 4*c2))/2
#         lam = np.max([lam1, lam2])
#         if (x/a)**2 + (y/b)**2 + (z/c)**2 >= 1.0:
#             u, v, w = flow_field(x, y, z, lam)
#             velocity_xz[j,i] = np.sqrt(u**2 + v**2 + w**2)
#         else:
#             velocity_xz[j,i] = np.nan

# for i in range(nx):
#     for j in range(ny):
#         x, y, z = X_xy[j,i], Y_xy[j,i], Z0[j,i]
#         c1 = a*a + b*b - x*x - y*y
#         c2 = a*a*b*b - b*b*x*x - a*a*y*y
#         lam1 = (-c1 + np.sqrt(c1*c1 - 4*c2))/2
#         lam2 = (-c1 - np.sqrt(c1*c1 - 4*c2))/2
#         lam = np.max([lam1, lam2])
#         if (x/a)**2 + (y/b)**2 + (z/c)**2 >= 1.0:
#             u, v, w = flow_field(x, y, z, lam)
#             velocity_xy[j,i] = np.sqrt(u**2 + v**2 + w**2)
#         else:
#             velocity_xy[j,i] = np.nan

# for i in range(ny):
#     for j in range(nz):
#         x, y, z = X0[j,i], Y_yz[j,i], Z_yz[j,i]
#         c1 = c*c + b*b - z*z - y*y
#         c2 = c*c*b*b - b*b*z*z - c*c*y*y
#         lam1 = (-c1 + np.sqrt(c1*c1 - 4*c2))/2
#         lam2 = (-c1 - np.sqrt(c1*c1 - 4*c2))/2
#         lam = np.max([lam1, lam2])
#         if (x/a)**2 + (y/b)**2 + (z/c)**2 >= 1.0:
#             u, v, w = flow_field(x, y, z, lam)
#             velocity_yz[j,i] = np.sqrt(u**2 + v**2 + w**2)
#         else:
#             velocity_yz[j,i] = np.nan

# end_time = time.time()

# print(f"runtime:{end_time - start_time:.2f}")


# fig, axes = plt.subplots(1, 3, figsize=(15,10))

# vmin = min(np.min(velocity_xy), np.min(velocity_xz), np.min(velocity_yz))
# vmax = max(np.max(velocity_xy), np.max(velocity_xz), np.max(velocity_yz))

# # x-y slice (z=0)
# im0 = axes[0].contourf(X_xy, Y_xy, velocity_xy, levels=50, cmap='viridis')
# axes[0].set_title('Velocity Magnitude (z=0 slice)')
# axes[0].set_xlabel('x')
# axes[0].set_ylabel('y')
# axes[0].axis('equal')

# # x-z slice (y=0)
# im1 = axes[1].contourf(X_xz, Z_xz, velocity_xz, levels=50, cmap='viridis')
# axes[1].set_title('Velocity Magnitude (y=0 slice)')
# axes[1].set_xlabel('x')
# axes[1].set_ylabel('z')
# axes[1].axis('equal')

# # y-z slice (x=0)
# im2 = axes[2].contourf(Y_yz, Z_yz, velocity_yz, levels=50, cmap='viridis')
# axes[2].set_title('Velocity Magnitude (x=0 slice)')
# axes[2].set_xlabel('y')
# axes[2].set_ylabel('z')
# axes[2].axis('equal')

# fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.046, pad=0.05)

# plt.show()

#endregion

#region stress

# Surface normal 

def surface_normal(x, y, z):
    n = np.array([x/a**2, y/b**2, z/c**2])
    return n / np.linalg.norm(n)

def velocity_gradient(x, y, z, lam, h=1e-5):
    u0, v0, w0 = flow_field(x, y, z, lam)

    ux = (flow_field(x+h, y, z, lam)[0] - u0) / h
    uy = (flow_field(x, y+h, z, lam)[0] - u0) / h
    uz = (flow_field(x, y, z+h, lam)[0] - u0) / h

    vx = (flow_field(x+h, y, z, lam)[1] - v0) / h
    vy = (flow_field(x, y+h, z, lam)[1] - v0) / h
    vz = (flow_field(x, y, z+h, lam)[1] - v0) / h

    wx = (flow_field(x+h, y, z, lam)[2] - w0) / h
    wy = (flow_field(x, y+h, z, lam)[2] - w0) / h
    wz = (flow_field(x, y, z+h, lam)[2] - w0) / h

    return np.array([
        [ux, uy, uz],
        [vx, vy, vz],
        [wx, wy, wz]
    ])

def stress_tensor(x, y, z, lam):
    grad_u = velocity_gradient(x, y, z, lam)
    return mu*(grad_u + grad_u.T)

def traction(x, y, z, lam):
    n = surface_normal(x, y, z)
    sigma_v = stress_tensor(x, y, z, lam)
    return sigma_v @ n

def surface_shear_stress(x, y, z, lam):
    n = surface_normal(x, y, z)
    t = traction(x, y, z, lam)
    normal_comp = np.dot(t,n) *n
    shear_comp = t -normal_comp
    return np.linalg.norm(shear_comp)

# n = 100
# x_vals = np.linspace(-5, 5, n)
# y_vals = np.linspace(-5, 5, n)
# z_vals = np.linspace(-5, 5, n)

# data = []

def compute_shear(x, y, z, lam=0):
    return surface_shear_stress(x,y,z,lam)

# z = 0
# shear_xy = np.zeros((n,n))
# for i, x in enumerate(x_vals):
#     for j, y in enumerate(y_vals):
#         if (x/a)**2 + (y/b)**2 + (z/c)**2 >= 1.0:
#             s = compute_shear(x, y, z)
#             shear_xy[j,i] = s
#             data.append([x, y, z, s])
#         else:
#             shear_xy[j,i] = np.nan

# y = 0
# shear_xz = np.zeros((n,n))
# for i, x in enumerate(x_vals):
#     for j, z in enumerate(z_vals):
#         if (x/a)**2 + (y/b)**2 + (z/c)**2 >= 1.0:
#             s = compute_shear(x, y, z)
#             shear_xz[j,i] = s
#             data.append([x, y, z, s])
#         else:
#             shear_xz[j,i] = np.nan

# x = 0
# shear_yz = np.zeros((n,n))
# for i, y in enumerate(y_vals):
#     for j, z in enumerate(z_vals):
#         if (x/a)**2 + (y/b)**2 + (z/c)**2 >= 1.0:
#             s = compute_shear(x, y, z)
#             shear_yz[j,i] = s
#             data.append([x, y, z, s])
#         else:
#             shear_yz[j,i] = np.nan

# with open('ellipsoid_shear.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['x','y','z','shear_stress'])
#     writer.writerows(data)

# print("Saved to ellipsoid_shear.csv")

# fig, axes = plt.subplots(1,3, figsize=(18,5))

# im0 = axes[0].contourf(x_vals, y_vals, shear_xy, levels=50, cmap='viridis')
# axes[0].set_title('x-y slice (z=0)')
# axes[0].set_xlabel('x'); axes[0].set_ylabel('y'); axes[0].axis('equal')

# im1 = axes[1].contourf(x_vals, z_vals, shear_xz, levels=50, cmap='viridis')
# axes[1].set_title('x-z slice (y=0)')
# axes[1].set_xlabel('x'); axes[1].set_ylabel('z'); axes[1].axis('equal')

# im2 = axes[2].contourf(y_vals, z_vals, shear_yz, levels=50, cmap='viridis')
# axes[2].set_title('y-z slice (x=0)')
# axes[2].set_xlabel('y'); axes[2].set_ylabel('z'); axes[2].axis('equal')

# fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.05, pad=0.05)
# plt.show()


n_theta = 400
theta = np.linspace(0, 2*np.pi, n_theta)

x_ell = a * np.sin(theta) # THIS HAS BEEN CHANGED - WAS X = A SIN THETA, Y = 0, Z = C COS THETA
y_ell = np.zeros_like(theta)
z_ell = c * np.cos(theta)

shear_surface = np.zeros(n_theta)

for i in range(n_theta):
    shear_surface[i] = compute_shear(x_ell[i], y_ell[i], z_ell[i])

dx_dtheta = a * np.cos(theta)
dz_dtheta =  -c * np.sin(theta)

ds_dtheta = np.sqrt(dx_dtheta**2 + dz_dtheta**2)
s = np.zeros(n_theta)
s[1:] = np.cumsum(0.5 * (ds_dtheta[1:] + ds_dtheta[:-1]) * np.diff(theta))

plt.figure(figsize=(7,4))
plt.plot(s, shear_surface, lw=2)
plt.xlabel("Arc length along ellipse")
plt.ylabel("Shear stress magnitude")
plt.title("Stress by arc length zx plane")
plt.grid(True)
plt.tight_layout()
plt.show()

#endregion

# analytical stress

shear_surface_a = np.zeros(n_theta)

def analytical_shear_stress(x,y,z, lam = 0):
    u_x = (a_bold + gamma_prime0*W - beta_prime0*V - 2*(alpha0 + beta0 + gamma0)*A + (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) 
            * (((-x*W-y*T)/(a*a*b*b) + (x*V-z*S)/(a*a*c*c) + (2*x*A-2*y*H-2*z*G)/(a*a) + (2*x*A+2*y*H)/(b*b) + (2*x*A+2*z*G)/(c*c))
            - (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(B-A)*y*y)/(b*b*b*b) + (2*(C-A)*z*z)/(c*c*c*c)
            )
            )
            ) 
    u_y = (h_bold - xi + gamma_prime0*T - (2*beta0-2*alpha0)*H + (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) 
            * (((-x*W-y*T)/(a*a*b*b) + (x*V-z*S)/(a*a*c*c) + (2*x*A-2*y*H-2*z*G)/(a*a) + (2*x*A+2*y*H)/(b*b) + (2*x*A+2*z*G)/(c*c))
            - (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(B-A)*y*y)/(b*b*b*b) + (2*(C-A)*z*z)/(c*c*c*c)
            )
            )
            )
    u_z = (g_bold + eta + beta_prime0*S - (2*gamma0 - 2*alpha0)*G + (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) 
            * (((-x*W-y*T)/(a*a*b*b) + (x*V-z*S)/(a*a*c*c) + (2*x*A-2*y*H-2*z*G)/(a*a) + (2*x*A+2*y*H)/(b*b) + (2*x*A+2*z*G)/(c*c))
            - (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(B-A)*y*y)/(b*b*b*b) + (2*(C-A)*z*z)/(c*c*c*c)
            )
            )
            ) 
    v_x = (h_bold + xi + gamma_prime0*T + (2*beta0-2*alpha0)*H + (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) 
            * (((-x*T+y*W)/(a*a*b*b) + (-y*U-z*R)/(b*b*c*c) + (2*x*H+2*y*B)/(a*a) + (-2*x*H+2*y*B-2*z*F)/(b*b) + (2*y*B+2*z*F)/(c*c))
            - (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(C-B)*z*z)/(c*c*c*c) + (2*(A-B)*x*x)/(a*a*a*a)
            )
            )
            ) 
    v_y = (b_bold + alpha_prime0*U - gamma_prime0*W - 2*(alpha0 + beta0 + gamma0)*B + (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) 
            * (((-x*T+y*W)/(a*a*b*b) + (-y*U-z*R)/(b*b*c*c) + (2*x*H+2*y*B)/(a*a) + (-2*x*H+2*y*B-2*z*F)/(b*b) + (2*y*B+2*z*F)/(c*c))
            - (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(C-B)*z*z)/(c*c*c*c) + (2*(A-B)*x*x)/(a*a*a*a)
            )
            )
            ) 
    v_z = (f_bold - xii +alpha_prime0*R - 2*gamma0*F + 2*beta0*F + (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) 
            * (((-x*T+y*W)/(a*a*b*b) + (-y*U-z*R)/(b*b*c*c) + (2*x*H+2*y*B)/(a*a) + (-2*x*H+2*y*B-2*z*F)/(b*b) + (2*y*B+2*z*F)/(c*c))
            - (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(C-B)*z*z)/(c*c*c*c) + (2*(A-B)*x*x)/(a*a*a*a)
            )
            )
            )     
    w_x = (g_bold - eta + beta_prime0*S + (2*gamma0 - 2*alpha0)*G + (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) 
            * (((z*U-y*R)/(c*c*b*b) + (-x*S-z*V)/(a*a*c*c) + (2*x*G+2*z*C)/(a*a) + (2*z*C+2*y*F)/(b*b) + (-2*x*G-2*y*F+2*z*C)/(c*c))
            - (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(A-C)*x*x)/(a*a*a*a) + (2*(B-C)*y*y)/(b*b*b*b)
            )
            )
            ) 
    w_y = (f_bold + xii + alpha_prime0*R + 2*gamma0*F -2*beta0*F + (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) 
            * (((z*U-y*R)/(c*c*b*b) + (-x*S-z*V)/(a*a*c*c) + (2*x*G+2*z*C)/(a*a) + (2*z*C+2*y*F)/(b*b) + (-2*x*G-2*y*F+2*z*C)/(c*c))
            - (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(A-C)*x*x)/(a*a*a*a) + (2*(B-C)*y*y)/(b*b*b*b)
            )
            )
            ) 
    w_z = (c_bold + beta_prime0*V - alpha_prime0*U - 2*(alpha0+beta0+gamma0)*C + (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) 
            * (((z*U-y*R)/(c*c*b*b) + (-x*S-z*V)/(a*a*c*c) + (2*x*G+2*z*C)/(a*a) + (2*z*C+2*y*F)/(b*b) + (-2*x*G-2*y*F+2*z*C)/(c*c))
            - (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(A-C)*x*x)/(a*a*a*a) + (2*(B-C)*y*y)/(b*b*b*b)
            )
            )
            ) 
    gradient = np.array([[u_x, u_y, u_z],[v_x, v_y, v_z],[w_x, w_y, w_z]])
    sigma = mu * (gradient +gradient.T)
    P0 = 1/(np.sqrt(((x*x)/(a*a*a*a)) + ((y*y)/(b*b*b*b)) + ((z*z)/(c*c*c*c))))
    normal = np.array([(P0*x)/(a*a), (P0*y)/(b*b), (P0*z)/(c*c)])
    traction = sigma @ normal 
    sss = traction - (np.dot(traction, normal)*normal)
    mag = np.linalg.norm(sss)
    return mag


for i in range(n_theta):
    shear_surface_a[i]= analytical_shear_stress(x_ell[i], y_ell[i], z_ell[i])

dx_dtheta = a * np.cos(theta)
dz_dtheta =  -c * np.sin(theta)

ds_dtheta = np.sqrt(dx_dtheta**2 + dz_dtheta**2)
s = np.zeros(n_theta)
s[1:] = np.cumsum(0.5 * (ds_dtheta[1:] + ds_dtheta[:-1]) * np.diff(theta))

plt.figure(figsize=(7,4))
plt.plot(s, shear_surface_a, lw=2)
plt.xlabel("Arc length along ellipse")
plt.ylabel("Shear stress magnitude")
plt.title("Analytical Version")
plt.grid(True)
plt.tight_layout()
plt.show()



# comparison of numerical vs analytical

# fig, axes = plt.subplots(1, 2, figsize=(12,4))

# axes[0].plot(s, shear_surface, lw=2)
# axes[0].set_xlabel("Arc length")
# axes[0].set_ylabel("Shear stress magnitude")
# axes[0].set_title("Numerical")
# axes[0].grid(True)

# axes[1].plot(s, shear_surface_a, lw=2)
# axes[1].set_xlabel("Arc length")
# axes[1].set_ylabel("Shear stress magnitude")
# axes[1].set_title("Analytical")
# axes[1].grid(True)

# plt.tight_layout()
# plt.show()

# difference

# plt.figure(figsize=(7,4))
# plt.plot(s, shear_surface - shear_surface_a, lw=2)
# plt.xlabel("Arc length")
# plt.ylabel("Difference")
# plt.title("Numerical - Analytical")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# validation

def pressure(x,y,z, lam=0):
     p = (2*mu*(2*A*alpha(lam) - (4*A*x*x*P_squared(x,y,z,lam))/((a*a + lam)**2 *Delta(lam))+ 
                2*B*beta(lam) - (4*B*y*y*P_squared(x,y,z,lam))/((b*b + lam)**2 *Delta(lam))+
                2*C*gamma(lam) - (4*C*z*z*P_squared(x,y,z,lam))/((c*c + lam)**2 *Delta(lam))-
                (8*F*y*z*P_squared(x,y,z,lam))/((b*b+lam)*(c*c+lam)*Delta(lam))-
                (8*G*z*x*P_squared(x,y,z,lam))/((a*a+lam)*(c*c+lam)*Delta(lam))-
                (8*H*x*y*P_squared(x,y,z,lam))/((a*a+lam)*(b*b+lam)*Delta(lam))
     )
     )
     return p 

def pressure_array(p):
    return np.array([[p, 0, 0], [0, p, 0], [0, 0, p]])

def analytical_grad_u(x,y,z, lam=0):
    u_x = (a_bold + gamma_prime0*W - beta_prime0*V - 2*(alpha0 + beta0 + gamma0)*A + (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) 
            * (((-x*W-y*T)/(a*a*b*b) + (x*V-z*S)/(a*a*c*c) + (2*x*A-2*y*H-2*z*G)/(a*a) + (2*x*A+2*y*H)/(b*b) + (2*x*A+2*z*G)/(c*c))
            - (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(B-A)*y*y)/(b*b*b*b) + (2*(C-A)*z*z)/(c*c*c*c)
            )
            )
            ) 
    u_y = (h_bold - xi + gamma_prime0*T - (2*beta0-2*alpha0)*H + (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) 
            * (((-x*W-y*T)/(a*a*b*b) + (x*V-z*S)/(a*a*c*c) + (2*x*A-2*y*H-2*z*G)/(a*a) + (2*x*A+2*y*H)/(b*b) + (2*x*A+2*z*G)/(c*c))
            - (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(B-A)*y*y)/(b*b*b*b) + (2*(C-A)*z*z)/(c*c*c*c)
            )
            )
            )
    u_z = (g_bold + eta + beta_prime0*S - (2*gamma0 - 2*alpha0)*G + (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) 
            * (((-x*W-y*T)/(a*a*b*b) + (x*V-z*S)/(a*a*c*c) + (2*x*A-2*y*H-2*z*G)/(a*a) + (2*x*A+2*y*H)/(b*b) + (2*x*A+2*z*G)/(c*c))
            - (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(B-A)*y*y)/(b*b*b*b) + (2*(C-A)*z*z)/(c*c*c*c)
            )
            )
            ) 
    v_x = (h_bold + xi + gamma_prime0*T + (2*beta0-2*alpha0)*H + (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) 
            * (((-x*T+y*W)/(a*a*b*b) + (-y*U-z*R)/(b*b*c*c) + (2*x*H+2*y*B)/(a*a) + (-2*x*H+2*y*B-2*z*F)/(b*b) + (2*y*B+2*z*F)/(c*c))
            - (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(C-B)*z*z)/(c*c*c*c) + (2*(A-B)*x*x)/(a*a*a*a)
            )
            )
            ) 
    v_y = (b_bold + alpha_prime0*U - gamma_prime0*W - 2*(alpha0 + beta0 + gamma0)*B + (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) 
            * (((-x*T+y*W)/(a*a*b*b) + (-y*U-z*R)/(b*b*c*c) + (2*x*H+2*y*B)/(a*a) + (-2*x*H+2*y*B-2*z*F)/(b*b) + (2*y*B+2*z*F)/(c*c))
            - (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(C-B)*z*z)/(c*c*c*c) + (2*(A-B)*x*x)/(a*a*a*a)
            )
            )
            ) 
    v_z = (f_bold - xii +alpha_prime0*R - 2*gamma0*F + 2*beta0*F + (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) 
            * (((-x*T+y*W)/(a*a*b*b) + (-y*U-z*R)/(b*b*c*c) + (2*x*H+2*y*B)/(a*a) + (-2*x*H+2*y*B-2*z*F)/(b*b) + (2*y*B+2*z*F)/(c*c))
            - (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(C-B)*z*z)/(c*c*c*c) + (2*(A-B)*x*x)/(a*a*a*a)
            )
            )
            )     
    w_x = (g_bold - eta + beta_prime0*S + (2*gamma0 - 2*alpha0)*G + (2*x*P_squared(x,y,z,lam))/(a*a*Delta(lam)) 
            * (((z*U-y*R)/(c*c*b*b) + (-x*S-z*V)/(a*a*c*c) + (2*x*G+2*z*C)/(a*a) + (2*z*C+2*y*F)/(b*b) + (-2*x*G-2*y*F+2*z*C)/(c*c))
            - (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(A-C)*x*x)/(a*a*a*a) + (2*(B-C)*y*y)/(b*b*b*b)
            )
            )
            ) 
    w_y = (f_bold + xii + alpha_prime0*R + 2*gamma0*F -2*beta0*F + (2*y*P_squared(x,y,z,lam))/(b*b*Delta(lam)) 
            * (((z*U-y*R)/(c*c*b*b) + (-x*S-z*V)/(a*a*c*c) + (2*x*G+2*z*C)/(a*a) + (2*z*C+2*y*F)/(b*b) + (-2*x*G-2*y*F+2*z*C)/(c*c))
            - (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(A-C)*x*x)/(a*a*a*a) + (2*(B-C)*y*y)/(b*b*b*b)
            )
            )
            ) 
    w_z = (c_bold + beta_prime0*V - alpha_prime0*U - 2*(alpha0+beta0+gamma0)*C + (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) 
            * (((z*U-y*R)/(c*c*b*b) + (-x*S-z*V)/(a*a*c*c) + (2*x*G+2*z*C)/(a*a) + (2*z*C+2*y*F)/(b*b) + (-2*x*G-2*y*F+2*z*C)/(c*c))
            - (2*z*P_squared(x,y,z,lam))/(c*c*Delta(lam)) * ((4*F*y*z)/(b*b*c*c) + (4*G*z*x)/(a*a*c*c) + (4*H*x*y)/(a*a*b*b) + (2*(A-C)*x*x)/(a*a*a*a) + (2*(B-C)*y*y)/(b*b*b*b)
            )
            )
            ) 
    gradient = np.array([[u_x, u_y, u_z],[v_x, v_y, v_z],[w_x, w_y, w_z]])
    return gradient

def analytical_stress_tensor(x,y,z, lam=0):
    grad_u = analytical_grad_u(x,y,z, lam)
    p = pressure(x,y,z, lam)
    return mu*(grad_u + grad_u.T) - pressure_array(p)

print("stress tensor" , analytical_stress_tensor(1, 1, 1))

def div_stress(x, y, z, lam=0):
    dx = 1e-5  

    sigma_xp = analytical_stress_tensor(x+dx, y, z, lam)
    sigma_xm = analytical_stress_tensor(x-dx, y, z, lam)
    dsigma_dx = (sigma_xp - sigma_xm) / (2*dx)

    sigma_yp = analytical_stress_tensor(x, y+dx, z, lam)
    sigma_ym = analytical_stress_tensor(x, y-dx, z, lam)
    dsigma_dy = (sigma_yp - sigma_ym) / (2*dx)

    sigma_zp = analytical_stress_tensor(x, y, z+dx, lam)
    sigma_zm = analytical_stress_tensor(x, y, z-dx, lam)
    dsigma_dz = (sigma_zp - sigma_zm) / (2*dx)

    div_x = dsigma_dx[0,0] + dsigma_dy[0,1] + dsigma_dz[0,2]
    div_y = dsigma_dx[1,0] + dsigma_dy[1,1] + dsigma_dz[1,2]
    div_z = dsigma_dx[2,0] + dsigma_dy[2,1] + dsigma_dz[2,2]

    return np.array([div_x, div_y, div_z])

print('component 1 of stress at (1,1,1):', div_stress(1, 1, 0)[0])
print('component 2 of stress at (1,1,1):', div_stress(1, 1, 0)[1])
print('component 3 of stress at (1,1,1):', div_stress(1, 1, 0)[2])


n_theta = 400
theta = np.linspace(0, 2*np.pi, n_theta)
x_ell = a * np.sin(theta)  
y_ell = np.zeros_like(theta)
z_ell = c * np.cos(theta) 
div_vals = np.zeros(n_theta)
div_mag_vals = np.zeros(n_theta)

for i in range(n_theta):
    div_vec = div_stress(x_ell[i], y_ell[i], z_ell[i])
    div_vals[i] = div_vec[0]  
    div_mag_vals[i] = np.linalg.norm(div_vec)
dx_dtheta = a * np.cos(theta)
dz_dtheta = -c * np.sin(theta)
ds_dtheta = np.sqrt(dx_dtheta**2 + dz_dtheta**2)
s = np.zeros(n_theta)
s[1:] = np.cumsum(0.5 * (ds_dtheta[1:] + ds_dtheta[:-1]) * np.diff(theta))
plt.plot(s, div_mag_vals)
plt.xlabel("Arc length along ellipse")
plt.ylabel("mag div sigma")
plt.title("should be 0")
plt.grid(True)
plt.show()









