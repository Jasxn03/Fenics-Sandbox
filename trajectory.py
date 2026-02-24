import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

U_tg = 1
dt = 0.01
steps = 5000


x = 0.01
y = 0.01

traj_x = []
traj_y = []

Lx = 2*np.pi
Ly = 2*np.pi


# def velocity(x, y):
#     u_x = U_tg*np.sin(x*2*np.pi/Lx)*np.cos(y*2*np.pi/Ly)
#     u_y = -U_tg*np.cos(x*2*np.pi/Lx)*np.sin(y*2*np.pi/Ly)
#     return u_x, u_y

def velocity(x,y):
    k = 0.5
    u_x = k*y
    u_y = 0
    return u_x, u_y 

for _ in range(steps):
    u_x, u_y = velocity(x, y)

    x += u_x * dt
    y += u_y * dt

    traj_x.append(x)
    traj_y.append(y)

plt.plot(traj_x, traj_y)
plt.title("Particle trajectory in Taylor–Green vortex")
plt.axis("equal")
plt.show()

mu = 1.0
sigma_mag = []

sigma_mag = []
for x, y in zip(traj_x, traj_y):
    # dux_dx = U_tg*np.cos(x*2*np.pi/Lx)*np.cos(y*2*np.pi/Ly) * 2 * np.pi/Lx
    # dux_dy = -U_tg*np.sin(x*2*np.pi/Lx)*np.sin(y*2*np.pi/Ly)* 2 * np.pi/Ly
    # duy_dx = U_tg*np.sin(x*2*np.pi/Lx)*np.sin(y*2*np.pi/Ly)* 2 * np.pi/Lx
    # duy_dy = -U_tg*np.cos(x*2*np.pi/Lx)*np.cos(y*2*np.pi/Ly)* 2 * np.pi/Ly

    dux_dx = 0
    dux_dy = 1
    duy_dx = 0
    duy_dy = 0

    grad = np.array([[dux_dx, dux_dy],
                     [duy_dx, duy_dy]])

    sigma = mu*(grad + grad.T)  
    sigma_mag.append(np.linalg.norm(sigma))

plt.plot(range(steps), sigma_mag)
plt.title("Stress mag at each step")
plt.show()


# region Jeffery

a = b = c = 0.01

def Delta(lam):
    return np.sqrt((a*a + lam)*(b*b + lam)*(c*c + lam))

def alpha(lam):
    f = lambda s: 1.0 / ((a*a + s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def beta(lam):
    f = lambda s: 1.0 / ((b*b + s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def gamma(lam):
    f = lambda s: 1.0 / ((c*c + s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def alpha_prime(lam):
    f = lambda s: 1.0 / ((b*b + s)*(c*c+s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def beta_prime(lam):
    f = lambda s: 1.0 / ((a*a + s)*(c*c +s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def gamma_prime(lam):
    f = lambda s: 1.0 / ((a*a + s)*(b*b + s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def alpha_prime2(lam):
    f = lambda s: s / ((b*b + s)*(c*c +s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def beta_prime2(lam):
    f = lambda s: s / ((a*a + s)*(c*c +s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def gamma_prime2(lam):
    f = lambda s: s / ((a*a + s)*(b*b + s)*Delta(s))
    return quad(f, lam, np.inf, limit=2000)[0]

def P2(x,y,z,lam):
    return 1/(((x*x)/(a*a + lam)**2) + ((y*y)/(b*b + lam)**2) + ((z*z)/(c*c + lam)**2))

def dl_dx(x,y,z,lam):
    _ = (2*x*P2(x,y,z,lam))/(a**2 + lam)
    return _ 

def dl_dy(x,y,z,lam):
    _ = (2*y*P2(x,y,z,lam))/(b**2 + lam)
    return _

def dl_dz(x,y,z,lam):
    _ = (2*z*P2(x,y,z,lam))/(c**2 + lam)
    return _

def F_coeff(x,y,z,lam):
    return ((x*x)/(a*a + lam)**2) + ((y*y)/(b*b + lam)**2) + ((z*z)/(c*c + lam)**2)

def P(x,y,z,lam):
    return F_coeff(x,y,z,lam)**(-1/2)

def dP_dx(x,y,z,lam):
    return -1/2 * (F_coeff(x,y,z,lam)**(-3/2)) * ((2*x)/(a**2 + lam) - (x**2 * dl_dx(x,y,z,lam))/(a**2 + lam)**2 - (y**2 * dl_dx(x,y,z,lam))/(b**2 + lam)**2 - (z**2 * dl_dx(x,y,z,lam))/(c**2 + lam)**2)
def dP_dy(x,y,z,lam):
    return -1/2 * (F_coeff(x,y,z,lam)**(-3/2)) * ((2*y)/(b**2 + lam) - (x**2 * dl_dy(x,y,z,lam))/(a**2 + lam)**2 - (y**2 * dl_dy(x,y,z,lam))/(b**2 + lam)**2 - (z**2 * dl_dy(x,y,z,lam))/(c**2 + lam)**2)
def dP_dz(x,y,z,lam):
    return -1/2 * (F_coeff(x,y,z,lam)**(-3/2)) * ((2*z)/(c**2 + lam) - (x**2 * dl_dz(x,y,z,lam))/(a**2 + lam)**2 - (y**2 * dl_dz(x,y,z,lam))/(b**2 + lam)**2 - (z**2 * dl_dz(x,y,z,lam))/(c**2 + lam)**2)
def dDeltaminus_dx(x,y,z,lam):
    return -1/2 * ((a**2 + lam)*(b**2 + lam)*(c**2 + lam))**(-3/2) * ((dl_dx(x,y,z,lam)*(b**2 + lam)*(c**2 + lam)) + ((a**2 + lam)*dl_dx(x,y,z,lam)*(c**2 + lam)) + ((a**2 + lam)*(b**2 + lam)*dl_dx(x,y,z,lam)))
def dDeltaminus_dy(x,y,z,lam):
    return -1/2 * ((a**2 + lam)*(b**2 + lam)*(c**2 + lam))**(-3/2) * ((dl_dy(x,y,z,lam)*(b**2 + lam)*(c**2 + lam)) + ((a**2 + lam)*dl_dy(x,y,z,lam)*(c**2 + lam)) + ((a**2 + lam)*(b**2 + lam)*dl_dy(x,y,z,lam)))
def dDeltaminus_dz(x,y,z,lam):
    return -1/2 * ((a**2 + lam)*(b**2 + lam)*(c**2 + lam))**(-3/2) * ((dl_dz(x,y,z,lam)*(b**2 + lam)*(c**2 + lam)) + ((a**2 + lam)*dl_dz(x,y,z,lam)*(c**2 + lam)) + ((a**2 + lam)*(b**2 + lam)*dl_dz(x,y,z,lam)))
def dalpha_dx(x,y,z,lam) :
    return -1/((a**2 + lam)*Delta(lam)) * dl_dx(x,y,z,lam)
def dbeta_dx(x,y,z,lam):
    return -1/((b**2 + lam)*Delta(lam)) * dl_dx(x,y,z,lam)
def dgamma_dx(x,y,z,lam):
    return -1/((c**2 + lam)*Delta(lam)) * dl_dx(x,y,z,lam)
def dalphaprime_dx(x,y,z,lam):
    return -1/((b**2 + lam)*(c**2+lam)*Delta(lam)) * dl_dx(x,y,z,lam)
def dbetaprime_dx(x,y,z,lam):
    return -1/((a**2 + lam)*(c**2+lam)*Delta(lam)) * dl_dx(x,y,z,lam)
def dgammaprime_dx(x,y,z,lam):
    return -1/((a**2 + lam)*(b**2+lam)*Delta(lam)) * dl_dx(x,y,z,lam)
def dalpha_dy(x,y,z,lam):
    return -1/((a**2 + lam)*Delta(lam)) * dl_dy(x,y,z,lam)
def dbeta_dy(x,y,z,lam):
    return -1/((b**2 + lam)*Delta(lam)) * dl_dy(x,y,z,lam)
def dgamma_dy(x,y,z,lam):
    return -1/((c**2 + lam)*Delta(lam)) * dl_dy(x,y,z,lam)
def dalphaprime_dy(x,y,z,lam):
    return -1/((b**2 + lam)*(c**2+lam)*Delta(lam)) * dl_dy(x,y,z,lam)
def dbetaprime_dy(x,y,z,lam):
    return -1/((a**2 + lam)*(c**2+lam)*Delta(lam)) * dl_dy(x,y,z,lam)
def dgammaprime_dy(x,y,z,lam):
    return -1/((a**2 + lam)*(b**2+lam)*Delta(lam)) * dl_dy(x,y,z,lam)
def dalpha_dz(x,y,z,lam):
    return -1/((a**2 + lam)*Delta(lam)) * dl_dz(x,y,z,lam)
def dbeta_dz(x,y,z,lam):
    return -1/((b**2 + lam)*Delta(lam)) * dl_dz(x,y,z,lam)
def dgamma_dz(x,y,z,lam):
    return -1/((c**2 + lam)*Delta(lam)) * dl_dz(x,y,z,lam)
def dalphaprime_dz(x,y,z,lam):
    return -1/((b**2 + lam)*(c**2+lam)*Delta(lam)) * dl_dz(x,y,z,lam)
def dbetaprime_dz(x,y,z,lam):
    return -1/((a**2 + lam)*(c**2+lam)*Delta(lam)) * dl_dz(x,y,z,lam)
def dgammaprime_dz(x,y,z,lam):
    return -1/((a**2 + lam)*(b**2+lam)*Delta(lam)) * dl_dz(x,y,z,lam)

# axis m_i and n_i (assumed that ellipsoid is aligned with the flow which makes sense for me?) og: m1 = 1 , n2 =1
m_1 = 0
n_1 = 1
m_2 = 1
n_2 = 0
m_3 = 0
n_3 = 0


def kappa(x,y,z=0):
    #return U_tg * np.sqrt(np.sin(x*2*np.pi/Lx)**2*np.cos(y*2*np.pi/Ly)**2 + np.cos(x*2*np.pi/Lx)**2*np.sin(y*2*np.pi/Ly)**2) 
    return 1
# flow field coefficients
def a_bold(x,y,z):
    #return kappa(x,y,z) * m_1 * n_1
    #return U_tg * np.cos(x*2*np.pi/Lx) * np.cos(y*2*np.pi/Ly)
    return 0
def b_bold(x,y,z):
    #return kappa(x,y,z) * m_2 * n_2
    #return -U_tg * np.cos(x*2*np.pi/Lx) * np.cos(y*2*np.pi/Ly)
    return 0 
def c_bold(x,y,z):
    return kappa(x,y,z) * m_3 * n_3
def f_bold(x,y,z):
    return 1/2 * kappa(x,y,z) * (m_2 * n_3 + m_3 * n_2)
def g_bold(x,y,z):
    return 1/2 * kappa(x,y,z) * (m_3 * n_1 + m_1 * n_3)
def h_bold(x,y,z):
    return 1/2 * kappa(x,y,z) * (m_1 * n_2 + m_2 * n_1)
def xii(x,y,z):
    return 1/2 * kappa(x,y,z) * (m_2 * n_3 - m_3 * n_2)
def eta(x,y,z):
    return 1/2 * kappa(x,y,z) * (m_3 * n_1 - m_1 * n_3)
def xi(x,y,z):
    #return -1/2 * kappa(x,y,z) * (m_1 * n_2 - m_2 * n_1)
    #return -U_tg * np.sin(x*2*np.pi/Lx) * np.sin(y*2*np.pi/Ly)
    return 0


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


def A(x,y,z):
    return 1/6 * (2*alpha_prime2_0* a_bold(x,y,z) - beta_prime2_0*b_bold(x,y,z) - gamma_prime2_0*c_bold(x,y,z))/(beta_prime2_0*gamma_prime2_0 + gamma_prime2_0*alpha_prime2_0+alpha_prime2_0*beta_prime2_0)
def B(x,y,z):
    return 1/6 * (2*beta_prime2_0* b_bold(x,y,z) - alpha_prime2_0*a_bold(x,y,z) - gamma_prime2_0*c_bold(x,y,z))/(beta_prime2_0*gamma_prime2_0 + gamma_prime2_0*alpha_prime2_0+alpha_prime2_0*beta_prime2_0)
def C(x,y,z):
    return 1/6 * (2*gamma_prime2_0* c_bold(x,y,z) - beta_prime2_0*b_bold(x,y,z) - alpha_prime2_0*a_bold(x,y,z))/(beta_prime2_0*gamma_prime2_0 + gamma_prime2_0*alpha_prime2_0+alpha_prime2_0*beta_prime2_0)
def F(x,y,z):
    return f_bold(x,y,z)/(2*alpha_prime0*(b*b + c*c))
def G(x,y,z):
    return g_bold(x,y,z)/(2*beta_prime0*(a*a + c*c))
def H(x,y,z):
    return h_bold(x,y,z)/(2*gamma_prime0*(b*b + a*a))
def R(x,y,z):
    return -f_bold(x,y,z)/alpha_prime0
def S(x,y,z):
    return -g_bold(x,y,z)/beta_prime0
def T(x,y,z):
    return -h_bold(x,y,z)/gamma_prime0
def U(x,y,z):
    return 2*b*b*B(x,y,z) - 2*c*c*C(x,y,z)
def V(x,y,z):
    return 2*c*c*C(x,y,z) - 2*a*a*A(x,y,z)
def W(x,y,z):
    return 2*a*a*A(x,y,z) - 2*b*b*B(x,y,z)

def du_dx(x,y,z,lam):
    u_x = (a_bold(x,y,z) + gamma_prime(lam)*W(x,y,z) - beta_prime(lam)*V(x,y,z) - 2*(alpha(lam)+beta(lam)+gamma(lam))*A(x,y,z) + x*W(x,y,z)*dgammaprime_dx(x,y,z,lam) -x*V(x,y,z)*dbetaprime_dx(x,y,z,lam) - 2*A(x,y,z)*x*(dalpha_dx(x,y,z,lam) + dbeta_dx(x,y,z,lam) + dgamma_dx(x,y,z,lam)) 
        + y*T(x,y,z)*dgammaprime_dx(x,y,z,lam) - 2*y*H(x,y,z)*(dbeta_dx(x,y,z,lam) - dalpha_dx(x,y,z,lam)) + z*S(x,y,z)*dbetaprime_dx(x,y,z,lam) - 2*z*G(x,y,z)*(dgamma_dx(x,y,z,lam)-dalpha_dx(x,y,z,lam))
        + ((-2*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam)) - (4*x*P(x,y,z,lam)*dP_dx(x,y,z,lam))/((a**2 + lam)*Delta(lam)) + (2*x*P2(x,y,z,lam)*dl_dx(x,y,z,lam))/((a**2 +lam)**2*Delta(lam)) - (2*x*P2(x,y,z,lam)*dDeltaminus_dx(x,y,z,lam))/(a**2 + lam))
        * ((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam))
            + ((2*lam*B(x,y,z)-2*lam*A(x,y,z))*y*y)/((b*b+lam)*(b*b+lam)) + ((2*lam*C(x,y,z) - 2*lam*A(x,y,z))*z*z)/((c*c+lam)*(c*c+lam)))
        - (2*x*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam))
        * ((4*dl_dx(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dx(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dx(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G(x,y,z)*z)/((c*c+lam)*(a*a+lam)) + (4*dl_dx(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dx(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dx(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H(x,y,z)*y)/((a*a+lam)*(b*b+lam)) + (4*dl_dx(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dx(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dx(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*B(x,y,z)-2*A(x,y,z))*y*y*dl_dx(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*B(x,y,z)-2*lam*A(x,y,z))*y*y)/((b*b+lam)**3)
            + ((2*C(x,y,z) - 2*A(x,y,z))*z*z*dl_dx(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*C(x,y,z) - 2*lam*A(x,y,z))*z*z)/((c*c+lam)**3))
        )
    return u_x 


def du_dy(x,y,z,lam):
    u_y = (x*W(x,y,z)*dgammaprime_dy(x,y,z,lam) -x*V(x,y,z)*dbetaprime_dy(x,y,z,lam)-2*A(x,y,z)*x*(dalpha_dy(x,y,z,lam)+ dbeta_dy(x,y,z,lam) + dgamma_dy(x,y,z,lam)) + h_bold(x,y,z) - xi(x,y,z) + gamma_prime(lam)*T(x,y,z) - 2*H(x,y,z)*(beta(lam) - alpha(lam)) + T(x,y,z)*y*dgammaprime_dy(x,y,z,lam) 
            - 2*y*H(x,y,z)*(dbeta_dy(x,y,z,lam) - dalpha_dy(x,y,z,lam)) + z*S(x,y,z)*dbetaprime_dy(x,y,z,lam) - 2*G(x,y,z)*z*(dgamma_dy(x,y,z,lam) - dalpha_dy(x,y,z,lam))
            + (- (4*x*P(x,y,z,lam)*dP_dy(x,y,z,lam))/((a**2 + lam)*Delta(lam)) + (2*x*P2(x,y,z,lam)*dl_dy(x,y,z,lam))/((a**2 +lam)**2*Delta(lam)) - (2*x*P2(x,y,z,lam)*dDeltaminus_dy(x,y,z,lam))/(a**2 + lam))
            * ((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam))
                + ((2*lam*B(x,y,z)-2*lam*A(x,y,z))*y*y)/((b*b+lam)*(b*b+lam)) + ((2*lam*C(x,y,z) - 2*lam*A(x,y,z))*z*z)/((c*c+lam)*(c*c+lam)))
            - (2*x*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam))
            * ((4*F(x,y,z)*z*lam)/((b**2 + lam)*(c**2 + lam)) + (4*dl_dy(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dy(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dy(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
                + (4*dl_dy(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dy(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dy(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
                + (4*lam*H(x,y,z)*x)/((a*a+lam)*(b*b+lam)) + (4*dl_dy(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dy(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dy(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
                + ((2*B(x,y,z)-2*A(x,y,z))*y*y*dl_dy(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*B(x,y,z)-2*lam*A(x,y,z))*y*y)/((b*b+lam)**3) + (2*(2*lam*B(x,y,z) - 2*lam*A(x,y,z))*y)/((b**2+lam)**2)
                + ((2*C(x,y,z) - 2*A(x,y,z))*z*z*dl_dy(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*C(x,y,z) - 2*lam*A(x,y,z))*z*z)/((c*c+lam)**3))
            )
    return u_y
def du_dz(x,y,z,lam):
    u_z = (x*W(x,y,z)*dgammaprime_dz(x,y,z,lam) - x*V(x,y,z)*dbetaprime_dz(x,y,z,lam) -2*A(x,y,z)*(dalpha_dz(x,y,z,lam) + dbeta_dz(x,y,z,lam) + dgamma_dz(x,y,z,lam)) + y*T(x,y,z)*dgammaprime_dz(x,y,z,lam) - 2*y*H(x,y,z)*(dbeta_dz(x,y,z,lam) - dalpha_dz(x,y,z,lam)) + g_bold(x,y,z) + eta(x,y,z) + beta_prime(lam)*S(x,y,z)
            - 2*(gamma(lam) - alpha(lam))*G(x,y,z) + z*S(x,y,z)*dbetaprime_dz(x,y,z,lam) - 2*z*G(x,y,z)*(dgamma_dz(x,y,z,lam) - dalpha_dz(x,y,z,lam))
            + (- (4*x*P(x,y,z,lam)*dP_dz(x,y,z,lam))/((a**2 + lam)*Delta(lam)) + (2*x*P2(x,y,z,lam)*dl_dz(x,y,z,lam))/((a**2 +lam)**2*Delta(lam)) - (2*x*P2(x,y,z,lam)*dDeltaminus_dz(x,y,z,lam))/(a**2 + lam))
            * ((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam))
                + ((2*lam*B(x,y,z)-2*lam*A(x,y,z))*y*y)/((b*b+lam)*(b*b+lam)) + ((2*lam*C(x,y,z) - 2*lam*A(x,y,z))*z*z)/((c*c+lam)*(c*c+lam)))
            - (2*x*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam))
            * ((4*lam*F(x,y,z)*y)/((b**2 + lam)*(c**2 + lam)) + (4*dl_dz(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dz(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dz(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
                + (4*lam*G(x,y,z)*x)/((c*c+lam)*(a*a+lam)) + (4*dl_dz(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dz(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dz(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
                + (4*dl_dz(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dz(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dz(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
                + ((2*B(x,y,z)-2*A(x,y,z))*y*y*dl_dz(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*B(x,y,z)-2*lam*A(x,y,z))*y*y)/((b*b+lam)**3)
                + ((2*C(x,y,z)-2*A(x,y,z))*z*z*dl_dz(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*C(x,y,z)-2*lam*A(x,y,z))*z*z)/((c*c+lam)**3) + (2*(2*lam*C(x,y,z) - 2*lam*A(x,y,z))*z)/((c**2+lam)**2))
            )
    return u_z

def dv_dx(x,y,z,lam):
    v_x = (h_bold(x,y,z) + xi(x,y,z) + gamma_prime(lam)*T(x,y,z) + 2*H(x,y,z)*(beta(lam) - alpha(lam)) + x*T(x,y,z)*dgammaprime_dx(x,y,z,lam) + 2*x*H(x,y,z)*(dbeta_dx(x,y,z,lam) - dalpha_dx(x,y,z,lam)) + y*U(x,y,z)*dalphaprime_dx(x,y,z,lam) - y*W(x,y,z)*dgammaprime_dx(x,y,z,lam)
        -2*y*B(x,y,z)*(dalpha_dx(x,y,z,lam) + dbeta_dx(x,y,z,lam) + dgamma_dx(x,y,z,lam)) + z*R(x,y,z)*dalphaprime_dx(x,y,z,lam) - 2*z*F(x,y,z)*(dgamma_dx(x,y,z,lam) - dbeta_dx(x,y,z,lam))
        +((-4*y*P(x,y,z,lam)*dP_dx(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (2*y*P2(x,y,z,lam)*dl_dx(x,y,z,lam))/((b**2+lam)**2*Delta(lam)) - (2*y*P2(x,y,z,lam) * dDeltaminus_dx(x,y,z,lam))/(b**2 + lam)) 
        *((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*C(x,y,z)-2*lam*B(x,y,z))*z*z)/((c*c+lam)*(c*c+lam))
        + ((2*lam*A(x,y,z) - 2*lam*B(x,y,z))*x*x)/((a*a+lam)*(a*a+lam)))
        - (2*y*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam))
        * ((4*dl_dx(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dx(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dx(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G(x,y,z)*z)/((c*c+lam)*(a*a+lam)) + (4*dl_dx(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dx(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dx(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H(x,y,z)*y)/((a**2+lam)*(b**2+lam)) + (4*dl_dx(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dx(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dx(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*C(x,y,z)-2*B(x,y,z))*z*z*dl_dx(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*C(x,y,z)-2*lam*B(x,y,z))*z*z)/((c*c+lam)**3)
            + ((2*A(x,y,z)-2*B(x,y,z))*x*x*dl_dx(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*A(x,y,z)-2*lam*B(x,y,z))*x*x)/((a*a+lam)**3) + (2*(2*lam*A(x,y,z) - 2*lam*B(x,y,z))*x)/((a**2+lam)**2))
        )
    return v_x

def dv_dy(x,y,z,lam):
    v_y = (x*T(x,y,z)*dgammaprime_dy(x,y,z,lam) + 2*x*H(x,y,z)*(dbeta_dy(x,y,z,lam) - dalpha_dy(x,y,z,lam)) + b_bold(x,y,z) + alpha_prime(lam)*U(x,y,z) - gamma_prime(lam)*W(x,y,z) - 2*(alpha(lam) + beta(lam) + gamma(lam))*B(x,y,z) + y*U(x,y,z)*dalphaprime_dy(x,y,z,lam) - y*W(x,y,z)*dgammaprime_dy(x,y,z,lam)
        -2*y*B(x,y,z)*(dalpha_dy(x,y,z,lam) + dbeta_dy(x,y,z,lam) + dgamma_dy(x,y,z,lam)) + z*R(x,y,z)*dalphaprime_dy(x,y,z,lam) - 2*z*F(x,y,z)*(dgamma_dy(x,y,z,lam) - dbeta_dy(x,y,z,lam))
        +((-2*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (-4*y*P(x,y,z,lam)*dP_dx(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (2*y*P2(x,y,z,lam)*dl_dy(x,y,z,lam))/((b**2+lam)**2*Delta(lam)) - (2*y*P2(x,y,z,lam) * dDeltaminus_dy(x,y,z,lam))/(b**2 + lam)) 
        *((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*C(x,y,z)-2*lam*B(x,y,z))*z*z)/((c*c+lam)*(c*c+lam))
        + ((2*lam*A(x,y,z) - 2*lam*B(x,y,z))*x*x)/((a*a+lam)*(a*a+lam)))
        - (2*y*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam))
        * ((4*lam*F(x,y,z)*z)/((b**2 + lam)*(c**2 + lam)) + (4*dl_dy(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dy(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dy(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*dl_dy(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dy(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dy(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H(x,y,z)*x)/((a**2+lam)*(b**2+lam)) + (4*dl_dy(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dy(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dy(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*C(x,y,z)-2*B(x,y,z))*z*z*dl_dy(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*C(x,y,z)-2*lam*B(x,y,z))*z*z)/((c*c+lam)**3)
            + ((2*A(x,y,z)-2*B(x,y,z))*x*x*dl_dy(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*A(x,y,z)-2*lam*B(x,y,z))*x*x)/((a*a+lam)**3))
        )
    return v_y 

def dv_dz(x,y,z,lam):
    v_z = (x*T(x,y,z)*dgammaprime_dz(x,y,z,lam) + 2*x*H(x,y,z)*(dbeta_dz(x,y,z,lam) - dalpha_dz(x,y,z,lam)) + y*U(x,y,z)*dalphaprime_dz(x,y,z,lam) - y*W(x,y,z)*dgammaprime_dz(x,y,z,lam) - 2*y*B(x,y,z)*(dalpha_dz(x,y,z,lam) + dbeta_dz(x,y,z,lam) + dgamma_dz(x,y,z,lam)) + f_bold(x,y,z) - xii(x,y,z) + alpha_prime(lam)*R(x,y,z)
        - (2*gamma(lam) - 2*beta(lam))*F(x,y,z) + z*R(x,y,z)*dalphaprime_dz(x,y,z,lam) - 2*z*F(x,y,z)*(dgamma_dz(x,y,z,lam) - dbeta_dz(x,y,z,lam))
        +((-4*y*P(x,y,z,lam)*dP_dz(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (2*y*P2(x,y,z,lam)*dl_dz(x,y,z,lam))/((b**2+lam)**2*Delta(lam)) - (2*y*P2(x,y,z,lam) * dDeltaminus_dz(x,y,z,lam))/(b**2 + lam)) 
        *((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*C(x,y,z)-2*lam*B(x,y,z))*z*z)/((c*c+lam)*(c*c+lam))
        + ((2*lam*A(x,y,z) - 2*lam*B(x,y,z))*x*x)/((a*a+lam)*(a*a+lam)))
        - (2*y*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam))
        * ((4*lam*F(x,y,z)*y)/((b**2+lam)*(c**2+lam)) + (4*dl_dz(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dz(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dz(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G(x,y,z)*x)/((c*c+lam)*(a*a+lam)) + (4*dl_dz(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dz(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dz(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*dl_dz(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dz(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dz(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*C(x,y,z)-2*B(x,y,z))*z*z*dl_dz(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*C(x,y,z)-2*lam*B(x,y,z))*z*z)/((c*c+lam)**3) + (2*(2*lam*C(x,y,z) - 2*lam*B(x,y,z))*z)/((c**2 + lam)**2)
            + ((2*A(x,y,z)-2*B(x,y,z))*x*x*dl_dz(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*A(x,y,z)-2*lam*B(x,y,z))*x*x)/((a*a+lam)**3))
        )
    return v_z

def dw_dx(x,y,z,lam):
    w_x = (g_bold(x,y,z)- eta(x,y,z) + beta_prime(lam)*S(x,y,z) - 2*(alpha(lam) -gamma(lam))*G(x,y,z) + x*S(x,y,z)*dbetaprime_dx(x,y,z,lam) - 2*x*G(x,y,z)*(dalpha_dx(x,y,z,lam) -dgamma_dx(x,y,z,lam)) + y*R(x,y,z)*dalphaprime_dx(x,y,z,lam) + 2*y*F(x,y,z)*(dgamma_dx(x,y,z,lam) - dbeta_dx(x,y,z,lam))
        + z*V(x,y,z)*dbetaprime_dx(x,y,z,lam) - z*U(x,y,z)*dalphaprime_dx(x,y,z,lam) - 2*z*C(x,y,z)*(dalpha_dx(x,y,z,lam) + dbeta_dx(x,y,z,lam) + dgamma_dx(x,y,z,lam))
        +((-4*z*P(x,y,z,lam)*dP_dx(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (2*z*P2(x,y,z,lam)*dl_dx(x,y,z,lam))/((c**2+lam)**2*Delta(lam)) - (2*z*P2(x,y,z,lam) * dDeltaminus_dx(x,y,z,lam))/(c**2 + lam))
        *((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*A(x,y,z)-2*lam*C(x,y,z))*x*x)/((a*a+lam)*(a*a+lam))
        + ((2*lam*B(x,y,z) - 2*lam*C(x,y,z))*y*y)/((b*b+lam)*(b*b+lam)))
        - (2*z*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam))
        * ((4*dl_dx(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dx(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dx(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G(x,y,z)*z)/((c*c+lam)*(a*a+lam)) + (4*dl_dx(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dx(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dx(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H(x,y,z)*y)/((a**2 + lam)*(b**2 + lam)) + (4*dl_dx(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dx(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dx(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*A(x,y,z)-2*C(x,y,z))*x*x*dl_dx(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*A(x,y,z)-2*lam*C(x,y,z))*x*x)/((a*a+lam)**3) + (2*(2*lam*A(x,y,z) - 2*lam*C(x,y,z))*x)/((a**2 + lam)**2)
            + ((2*B(x,y,z)-2*C(x,y,z))*y*y*dl_dx(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*B(x,y,z)-2*lam*C(x,y,z))*y*y)/((b*b+lam)**3)
            )
        )
    return w_x

def dw_dy(x,y,z,lam):
    w_y = (x*S(x,y,z)*dbetaprime_dy(x,y,z,lam) - 2*x*G(x,y,z)*(dalpha_dy(x,y,z,lam) - dgamma_dy(x,y,z,lam)) + f_bold(x,y,z) + xii(x,y,z) + alpha_prime(lam)*R(x,y,z) + 2*F(x,y,z)*(gamma(lam)-beta(lam)) + y*R(x,y,z)*dalphaprime_dy(x,y,z,lam) + 2*y*F(x,y,z)*(dgamma_dy (x,y,z,lam)- dbeta_dy(x,y,z,lam))
        + z*V(x,y,z)*dbetaprime_dy(x,y,z,lam)- z*U(x,y,z)*dalphaprime_dy(x,y,z,lam) -2*z*C(x,y,z)*(dalpha_dy(x,y,z,lam) + dbeta_dy(x,y,z,lam) +dgamma_dy(x,y,z,lam))
                +((-4*z*P(x,y,z,lam)*dP_dy(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (2*z*P2(x,y,z,lam)*dl_dy(x,y,z,lam))/((c**2+lam)**2*Delta(lam)) - (2*z*P2(x,y,z,lam) * dDeltaminus_dy(x,y,z,lam))/(c**2 + lam))
                *((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*A(x,y,z)-2*lam*C(x,y,z))*x*x)/((a*a+lam)*(a*a+lam))
                + ((2*lam*B(x,y,z) - 2*lam*C(x,y,z))*y*y)/((b*b+lam)*(b*b+lam)))
                - (2*z*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam))
                * ((4*lam*F(x,y,z)*z)/((b*b + lam)*(c*c + lam)) + (4*dl_dy(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dy(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dy(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
                    + (4*dl_dy(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dy(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dy(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
                    + (4*lam*H(x,y,z)*x)/((a**2 + lam)*(b**2 + lam)) + (4*dl_dy(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dy(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dy(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
                    + ((2*A(x,y,z)-2*C(x,y,z))*x*x*dl_dy(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*A(x,y,z)-2*lam*C(x,y,z))*x*x)/((a*a+lam)**3)
                    + ((2*B(x,y,z)-2*C(x,y,z))*y*y*dl_dy(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*B(x,y,z)-2*lam*C(x,y,z))*y*y)/((b*b+lam)**3) + (2*(2*lam*B(x,y,z) - 2*lam*C(x,y,z))*y)/((b**2 + lam)**2)
            )
        )
            
    return w_y

def dw_dz(x,y,z,lam):
    w_z = (x*S(x,y,z)*dbetaprime_dz(x,y,z,lam) - 2*x*G(x,y,z)*(dalpha_dz(x,y,z,lam) - dgamma_dz(x,y,z,lam)) + y*R(x,y,z)*dalphaprime_dz(x,y,z,lam) + 2*y*F(x,y,z)*(dgamma_dz(x,y,z,lam) - dbeta_dz(x,y,z,lam)) + c_bold(x,y,z) + beta_prime(lam)*V(x,y,z) -alpha_prime(lam)*U(x,y,z) -2*C(x,y,z)*(alpha(lam) + beta(lam) + gamma(lam))
        + z*V(x,y,z)*dbetaprime_dz(x,y,z,lam) -z*U(x,y,z)*dalphaprime_dz(x,y,z,lam) - 2*z*C(x,y,z)*(dalpha_dz(x,y,z,lam) + dbeta_dz(x,y,z,lam) +dgamma_dz(x,y,z,lam))
        +((-2*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (-4*z*P(x,y,z,lam)*dP_dz(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (2*z*P2(x,y,z,lam)*dl_dz(x,y,z,lam))/((c**2+lam)**2*Delta(lam)) - (2*z*P2(x,y,z,lam) * dDeltaminus_dz(x,y,z,lam))/(c**2 + lam))
        *((4*lam*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*A(x,y,z)-2*lam*C(x,y,z))*x*x)/((a*a+lam)*(a*a+lam))
        + ((2*lam*B(x,y,z) - 2*lam*C(x,y,z))*y*y)/((b*b+lam)*(b*b+lam)))
        - (2*z*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam))
        * ((4*lam*F(x,y,z)*y)/((b*b+lam)*(c*c+lam)) + (4*dl_dz(x,y,z,lam)*F(x,y,z)*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dz(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F(x,y,z)*y*z*dl_dz(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G(x,y,z)*x)/((c*c+lam)*(a*a+lam)) + (4*dl_dz(x,y,z,lam)*G(x,y,z)*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dz(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G(x,y,z)*z*x*dl_dz(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*dl_dz(x,y,z,lam)*H(x,y,z)*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dz(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H(x,y,z)*x*y*dl_dz(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*A(x,y,z)-2*C(x,y,z))*x*x*dl_dz(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*A(x,y,z)-2*lam*C(x,y,z))*x*x)/((a*a+lam)**3) 
            + ((2*B(x,y,z)-2*C(x,y,z))*y*y*dl_dz(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*B(x,y,z)-2*lam*C(x,y,z))*y*y)/((b*b+lam)**3)
            )
        )
    return w_z

    
def sigma_xx(x,y,z):
    lam = 0
    u_x = du_dx(x,y,z,lam)
    return 2*u_x

def sigma_xy(x,y,z):
    lam = 0
    u_y = du_dy(x,y,z,lam)
    v_x = dv_dx(x,y,z,lam)
    return u_y + v_x

def sigma_xz(x,y,z):
    lam = 0
    u_z = du_dz(x,y,z,lam)
    w_x = dw_dx(x,y,z,lam)
    return u_z + w_x

def sigma_yx(x,y,z):
    lam = 0
    v_x = dv_dx(x,y,z,lam)
    u_y = du_dy(x,y,z,lam)
    return v_x + u_y

def sigma_yy(x,y,z):
    lam = 0
    v_y = dv_dy(x,y,z,lam)
    return 2*v_y

def sigma_yz(x,y,z):
    lam = 0
    v_z = dv_dz(x,y,z,lam)
    w_y = dw_dy(x,y,z,lam)
    return v_z + w_y

def sigma_zx(x,y,z):
    lam = 0
    w_x = dw_dx(x,y,z,lam)
    u_z = du_dz(x,y,z,lam)
    return w_x + u_z

def sigma_zy(x,y,z):
    lam = 0
    w_y = dw_dy(x,y,z,lam)
    v_z = dv_dz(x,y,z,lam)
    return w_y + v_z

def sigma_zz(x,y,z):
    lam = 0
    w_z = dw_dz(x,y,z,lam)
    return 2*w_z

def to_surface(x_0, y_0, z_0=0):
    r = np.sqrt((x_0/a)**2 + (y_0/b)**2 + (z_0/c)**2)
    return x_0/r, y_0/r, z_0/r


sigma_mag_jeff = []
for x_0, y_0 in zip(traj_x, traj_y):
    z=0
    x = x_0
    y = y_0 - a 
    #x, y, z = to_surface(x_0, y_0, 0)

    s_xx = sigma_xx(x,y,z)
    s_xy = sigma_xy(x,y,z)
    s_yx = sigma_yx(x,y,z)
    s_yy = sigma_yy(x,y,z)
    sigma_mag_jeff.append(np.sqrt(s_xx**2 + s_xy**2 + s_yx**2 + s_yy**2))

plt.plot(range(steps), sigma_mag_jeff)
plt.title("Stress mag at each step")
plt.show()



plt.plot(range(steps), sigma_mag, label = "tg derived")
plt.plot(range(steps), sigma_mag_jeff, label = "jeff")
plt.legend()
plt.show()

print(f"ratio first {sigma_mag_jeff[0]/sigma_mag[0]}")
print(f"ratio last {sigma_mag_jeff[-1]/sigma_mag[-1]}")

#endregion
