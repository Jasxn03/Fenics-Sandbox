# kim and karrila sol

import numpy as np
import matplotlib.pyplot as plt


k = 1500       
a = 1.0       
mu = 1.0    

def velocity_field(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < a:
        return np.array([0.0, 0.0, 0.0])

    r2 = r**2
    r3 = r**3
    r5 = r**5

    ux = k*y*(1 - 5/2*(a**3/r3) + 3/2*(a**5*x**2/r5))
    uy = k*x*(- a**3/r3 + 3/2*a**5/r5)
    uz = k*x*z * 3/2 * a**5 / (r5)
    return np.array([ux, uy, uz])



def velocity_gradient(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < a:
        return 0.0
    r2 = r**2
    r3 = r**3
    r5 = r**5
    r7 = r**7

    u_x = k*y*(5/2 * a**3 * 3*x/r5 + 3/2 *a**5 * 2 * x/r5 - 3/2 *a**5 *x**2 * 5*x/r7)
    u_y = k*(1-5/2 * a**3 /r3 + 3/2 * a**5 * x**2 /r5) + k*y*(5/2 * a**3 *3*y/r5 -3/2 * a**5 * x**2 * 5 *y/r7)
    u_z = k*y*(5/2 * a**3 * 3 * z/r5 - 3/2 * a**5 * x**2 *5 * z/r7)

    v_x = k*(-a**3/r3 + 3/2*a**5/r5) + k*x*(a**3 * 3*x/r5 -3/2 * a**5 *5*x/r7)
    v_y = k*x*(a**3 * 3 * y/r5 -3/2 * a**5 *5*y/r7)
    v_z = k*x*(a**3 * 3 * z/r5 - 3/2 * a**5 * 5 * z/r7)

    w_x = 3*k*z*a**5/(2*r5) - 3*k*x*z*a**5/2 * 5 * x/r7
    w_y = -3*k*x*z*a**5/2 * 5*y/r7
    w_z = -3*k*x*z*a**5/2 * 5 *z/r7 + 3*k*x*a**5/(2*r5)

    gradient = np.array([[u_x, u_y, u_z],[v_x, v_y, v_z],[w_x, w_y, w_z]])
    normal = np.array([x/r, y/r, z/r])
    sigma = mu * (gradient +gradient.T)
    traction = sigma @ normal 
    sss = traction - (np.dot(traction, normal)*normal)
    mag = np.linalg.norm(sss)
    return mag

n_theta = 400
theta = np.linspace(0, 2*np.pi, n_theta)
shear_surface_a = np.zeros(n_theta)

# x_ell = a * np.sin(theta) 
# y_ell = np.zeros_like(theta)
# z_ell = a * np.cos(theta)

# x_ell = np.zeros_like(theta)
# y_ell = a * np.cos(theta)
# z_ell = a * np.sin(theta)

x_ell = a*np.sin(theta)
y_ell = a * np.cos(theta)
z_ell = np.zeros_like(theta)

dx_dtheta = a * np.cos(theta)
dz_dtheta =  -a * np.sin(theta)

for i in range(n_theta):
    shear_surface_a[i]= velocity_gradient(x_ell[i], y_ell[i], z_ell[i])


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
