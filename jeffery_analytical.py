import sympy as sp
from sympy.utilities.lambdify import lambdify
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import quad

start_time = time.time()

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
            - (2*x*P2(x,y,z,lam))/((a*a+lam) * Delta(lam)) * ((R + 2*(b*b+lam)*F + 2*(c*c+lam)*F) * y*z/((b*b+lam)*(c*c+lam))
                                            + (S + 2*(c*c+lam)*G + 2*(a*a+lam)*G) * z*x/((c*c+lam)*(a*a+lam))
                                            + (T + 2*(a*a + lam)*H + 2*(b*b + lam)*H) * x*y/((a*a+lam)*(b*b+lam))
                                            + (W- 2*(a*a+lam)*A + 2*(b*b+lam)*B) * y*y/((b*b+lam)*(b*b+lam))
                                            - (V - 2*(c*c+lam)*C + 2*(a*a+lam)*A) * z*z/((c*c+lam)*(c*c+lam))

            )
            )
    v =( x*(h_bold + xi + gamma_prime(lam)*T + 2*beta(lam)*H - 2*alpha(lam)*H) 
            + y*(b_bold - alpha_prime(lam)*U - gamma_prime(lam)*W - 2*(alpha(lam) + beta(lam) + gamma(lam))*B) 
            + z*(f_bold + xii + alpha_prime(lam)*R - 2*gamma(lam) * F + 2*beta(lam) * F)
            - (2*y*P2(x,y,z,lam))/((b*b+lam) * Delta(lam)) * ((R + 2*(b*b+lam)*F + 2*(c*c+lam)*F) * y*z/((b*b+lam)*(c*c+lam))
                                            + (S + 2*(c*c+lam)*G + 2*(a*a+lam)*G) * z*x/((c*c+lam)*(a*a+lam))
                                            + (T + 2*(a*a+lam)*H + 2*(b*b+lam)*H) * x*y/((a*a + lam)*(b*b+lam))
                                            + (U- 2*(b*b+lam)*B + 2*(c*c+lam)*C) * z*z/((c*c+lam)*(c*c+lam))
                                            - (W - 2*(a*a+lam)*A + 2*(b*b+lam)*B) * x*x/((a*a+lam)*(a*a+lam))

            )
            )
    w =( x*(g_bold - eta + beta_prime(lam)*S - 2*alpha(lam)*G + 2*gamma(lam) * G) 
            + y*(f_bold + xii + alpha_prime(lam)*R + 2*gamma(lam)*F - 2*beta(lam)*F) 
            + z*(c_bold + beta_prime(lam)*V - alpha_prime(lam)*U - 2*(alpha(lam) + beta(lam) + gamma(lam))*C)
            - (2*z*P2(x,y,z,lam))/((c*c+lam) * Delta(lam)) * ((R + 2*(b*b + lam)*F + 2*(c*c+lam)*F) * y*z/((b*b+lam)*(c*c+lam))
                                            + (S + 2*(c*c+lam)*G + 2*(a*a+lam)*G) * z*x/((c*c+lam)*(a*a+lam))
                                            + (T + 2*(a*a + lam)*H + 2*(b*b+lam)*H) * x*y/((a*a+lam)*(b*b+lam))
                                            + (V - 2*(c*c+lam)*C + 2*(a*a+lam)*A) * x*x/((a*a+lam)*(a*a+lam))
                                            - (U - 2*(b*b+lam)*B + 2*(c*c+lam)*C) * y*y/((b*b+lam)*(b*b+lam))

            )
            )
    return u ,v ,w

def du_dx(x,y,z,lam):
    u_x = (a_bold + gamma_prime(lam)*W - beta_prime(lam)*V - 2*(alpha(lam)+beta(lam)+gamma(lam))*A + x*W*dgammaprime_dx(x,y,z,lam) -x*V*dbetaprime_dx(x,y,z,lam) - 2*A*x*(dalpha_dx(x,y,z,lam) + dbeta_dx(x,y,z,lam) + dgamma_dx(x,y,z,lam)) 
        + y*T*dgammaprime_dx(x,y,z,lam) - 2*y*H*(dbeta_dx(x,y,z,lam) - dalpha_dx(x,y,z,lam)) + z*S*dbetaprime_dx(x,y,z,lam) - 2*z*G*(dgamma_dx(x,y,z,lam)-dalpha_dx(x,y,z,lam))
        + ((-2*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam)) - (4*x*P(x,y,z,lam)*dP_dx(x,y,z,lam))/((a**2 + lam)*Delta(lam)) + (2*x*P2(x,y,z,lam)*dl_dx(x,y,z,lam))/((a**2 +lam)**2*Delta(lam)) - (2*x*P2(x,y,z,lam)*dDeltaminus_dx(x,y,z,lam))/(a**2 + lam))
        * ((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam))
            + ((2*lam*B-2*lam*A)*y*y)/((b*b+lam)*(b*b+lam)) + ((2*lam*C - 2*lam*A)*z*z)/((c*c+lam)*(c*c+lam)))
        - (2*x*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam))
        * ((4*dl_dx(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dx(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dx(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G*z)/((c*c+lam)*(a*a+lam)) + (4*dl_dx(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dx(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dx(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H*y)/((a*a+lam)*(b*b+lam)) + (4*dl_dx(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dx(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dx(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*B-2*A)*y*y*dl_dx(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*B-2*lam*A)*y*y)/((b*b+lam)**3)
            + ((2*C - 2*A)*z*z*dl_dx(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*C - 2*lam*A)*z*z)/((c*c+lam)**3))
        )
    return u_x 


def du_dy(x,y,z,lam):
    u_y = (x*W*dgammaprime_dy(x,y,z,lam) -x*V*dbetaprime_dy(x,y,z,lam)-2*A*x*(dalpha_dy(x,y,z,lam)+ dbeta_dy(x,y,z,lam) + dgamma_dy(x,y,z,lam)) + h_bold - xi + gamma_prime(lam)*T - 2*H*(beta(lam) - alpha(lam)) + T*y*dgammaprime_dy(x,y,z,lam) 
            - 2*y*H*(dbeta_dy(x,y,z,lam) - dalpha_dy(x,y,z,lam)) + z*S*dbetaprime_dy(x,y,z,lam) - 2*G*z*(dgamma_dy(x,y,z,lam) - dalpha_dy(x,y,z,lam))
            + (- (4*x*P(x,y,z,lam)*dP_dy(x,y,z,lam))/((a**2 + lam)*Delta(lam)) + (2*x*P2(x,y,z,lam)*dl_dy(x,y,z,lam))/((a**2 +lam)**2*Delta(lam)) - (2*x*P2(x,y,z,lam)*dDeltaminus_dy(x,y,z,lam))/(a**2 + lam))
            * ((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam))
                + ((2*lam*B-2*lam*A)*y*y)/((b*b+lam)*(b*b+lam)) + ((2*lam*C - 2*lam*A)*z*z)/((c*c+lam)*(c*c+lam)))
            - (2*x*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam))
            * ((4*F*z*lam)/((b**2 + lam)*(c**2 + lam)) + (4*dl_dy(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dy(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dy(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
                + (4*dl_dy(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dy(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dy(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
                + (4*lam*H*x)/((a*a+lam)*(b*b+lam)) + (4*dl_dy(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dy(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dy(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
                + ((2*B-2*A)*y*y*dl_dy(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*B-2*lam*A)*y*y)/((b*b+lam)**3) + (2*(2*lam*B - 2*lam*A)*y)/((b**2+lam)**2)
                + ((2*C - 2*A)*z*z*dl_dy(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*C - 2*lam*A)*z*z)/((c*c+lam)**3))
            )
    return u_y
def du_dz(x,y,z,lam):
    u_z = (x*W*dgammaprime_dz(x,y,z,lam) - x*V*dbetaprime_dz(x,y,z,lam) -2*A*(dalpha_dz(x,y,z,lam) + dbeta_dz(x,y,z,lam) + dgamma_dz(x,y,z,lam)) + y*T*dgammaprime_dz(x,y,z,lam) - 2*y*H*(dbeta_dz(x,y,z,lam) - dalpha_dz(x,y,z,lam)) + g_bold + eta + beta_prime(lam)*S
            - 2*(gamma(lam) - alpha(lam))*G + z*S*dbetaprime_dz(x,y,z,lam) - 2*z*G*(dgamma_dz(x,y,z,lam) - dalpha_dz(x,y,z,lam))
            + (- (4*x*P(x,y,z,lam)*dP_dz(x,y,z,lam))/((a**2 + lam)*Delta(lam)) + (2*x*P2(x,y,z,lam)*dl_dz(x,y,z,lam))/((a**2 +lam)**2*Delta(lam)) - (2*x*P2(x,y,z,lam)*dDeltaminus_dz(x,y,z,lam))/(a**2 + lam))
            * ((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam))
                + ((2*lam*B-2*lam*A)*y*y)/((b*b+lam)*(b*b+lam)) + ((2*lam*C - 2*lam*A)*z*z)/((c*c+lam)*(c*c+lam)))
            - (2*x*P2(x,y,z,lam))/((a**2 + lam)*Delta(lam))
            * ((4*lam*F*y)/((b**2 + lam)*(c**2 + lam)) + (4*dl_dz(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dz(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dz(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
                + (4*lam*G*x)/((c*c+lam)*(a*a+lam)) + (4*dl_dz(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dz(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dz(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
                + (4*dl_dz(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dz(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dz(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
                + ((2*B-2*A)*y*y*dl_dz(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*B-2*lam*A)*y*y)/((b*b+lam)**3)
                + ((2*C-2*A)*z*z*dl_dz(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*C-2*lam*A)*z*z)/((c*c+lam)**3) + (2*(2*lam*C - 2*lam*A)*z)/((c**2+lam)**2))
            )
    return u_z

def dv_dx(x,y,z,lam):
    v_x = (h_bold + xi + gamma_prime(lam)*T + 2*H*(beta(lam) - alpha(lam)) + x*T*dgammaprime_dx(x,y,z,lam) + 2*x*H*(dbeta_dx(x,y,z,lam) - dalpha_dx(x,y,z,lam)) + y*U*dalphaprime_dx(x,y,z,lam) - y*W*dgammaprime_dx(x,y,z,lam)
        -2*y*B*(dalpha_dx(x,y,z,lam) + dbeta_dx(x,y,z,lam) + dgamma_dx(x,y,z,lam)) + z*R*dalphaprime_dx(x,y,z,lam) - 2*z*F*(dgamma_dx(x,y,z,lam) - dbeta_dx(x,y,z,lam))
        +((-4*y*P(x,y,z,lam)*dP_dx(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (2*y*P2(x,y,z,lam)*dl_dx(x,y,z,lam))/((b**2+lam)**2*Delta(lam)) - (2*y*P2(x,y,z,lam) * dDeltaminus_dx(x,y,z,lam))/(b**2 + lam)) 
        *((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*C-2*lam*B)*z*z)/((c*c+lam)*(c*c+lam))
        + ((2*lam*A - 2*lam*B)*x*x)/((a*a+lam)*(a*a+lam)))
        - (2*y*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam))
        * ((4*dl_dx(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dx(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dx(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G*z)/((c*c+lam)*(a*a+lam)) + (4*dl_dx(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dx(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dx(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H*y)/((a**2+lam)*(b**2+lam)) + (4*dl_dx(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dx(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dx(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*C-2*B)*z*z*dl_dx(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*C-2*lam*B)*z*z)/((c*c+lam)**3)
            + ((2*A-2*B)*x*x*dl_dx(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*A-2*lam*B)*x*x)/((a*a+lam)**3) + (2*(2*lam*A - 2*lam*B)*x)/((a**2+lam)**2))
        )
    return v_x

def dv_dy(x,y,z,lam):
    v_y = (x*T*dgammaprime_dy(x,y,z,lam) + 2*x*H*(dbeta_dy(x,y,z,lam) - dalpha_dy(x,y,z,lam)) + b_bold + alpha_prime(lam)*U - gamma_prime(lam)*W - 2*(alpha(lam) + beta(lam) + gamma(lam))*B + y*U*dalphaprime_dy(x,y,z,lam) - y*W*dgammaprime_dy(x,y,z,lam)
        -2*y*B*(dalpha_dy(x,y,z,lam) + dbeta_dy(x,y,z,lam) + dgamma_dy(x,y,z,lam)) + z*R*dalphaprime_dy(x,y,z,lam) - 2*z*F*(dgamma_dy(x,y,z,lam) - dbeta_dy(x,y,z,lam))
        +((-2*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (-4*y*P(x,y,z,lam)*dP_dy(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (2*y*P2(x,y,z,lam)*dl_dy(x,y,z,lam))/((b**2+lam)**2*Delta(lam)) - (2*y*P2(x,y,z,lam) * dDeltaminus_dy(x,y,z,lam))/(b**2 + lam)) 
        *((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*C-2*lam*B)*z*z)/((c*c+lam)*(c*c+lam))
        + ((2*lam*A - 2*lam*B)*x*x)/((a*a+lam)*(a*a+lam)))
        - (2*y*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam))
        * ((4*lam*F*z)/((b**2 + lam)*(c**2 + lam)) + (4*dl_dy(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dy(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dy(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*dl_dy(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dy(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dy(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H*x)/((a**2+lam)*(b**2+lam)) + (4*dl_dy(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dy(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dy(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*C-2*B)*z*z*dl_dy(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*C-2*lam*B)*z*z)/((c*c+lam)**3)
            + ((2*A-2*B)*x*x*dl_dy(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*A-2*lam*B)*x*x)/((a*a+lam)**3))
        )
    return v_y 

def dv_dz(x,y,z,lam):
    v_z = (x*T*dgammaprime_dz(x,y,z,lam) + 2*x*H*(dbeta_dz(x,y,z,lam) - dalpha_dz(x,y,z,lam)) + y*U*dalphaprime_dz(x,y,z,lam) - y*W*dgammaprime_dz(x,y,z,lam) - 2*y*B*(dalpha_dz(x,y,z,lam) + dbeta_dz(x,y,z,lam) + dgamma_dz(x,y,z,lam)) + f_bold - xii + alpha_prime(lam)*R
        - (2*gamma(lam) - 2*beta(lam))*F + z*R*dalphaprime_dz(x,y,z,lam) - 2*z*F*(dgamma_dz(x,y,z,lam) - dbeta_dz(x,y,z,lam))
        +((-4*y*P(x,y,z,lam)*dP_dz(x,y,z,lam))/((b**2 + lam)*Delta(lam)) + (2*y*P2(x,y,z,lam)*dl_dz(x,y,z,lam))/((b**2+lam)**2*Delta(lam)) - (2*y*P2(x,y,z,lam) * dDeltaminus_dz(x,y,z,lam))/(b**2 + lam)) 
        *((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*C-2*lam*B)*z*z)/((c*c+lam)*(c*c+lam))
        + ((2*lam*A - 2*lam*B)*x*x)/((a*a+lam)*(a*a+lam)))
        - (2*y*P2(x,y,z,lam))/((b**2 + lam)*Delta(lam))
        * ((4*lam*F*y)/((b**2+lam)*(c**2+lam)) + (4*dl_dz(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dz(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dz(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G*x)/((c*c+lam)*(a*a+lam)) + (4*dl_dz(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dz(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dz(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*dl_dz(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dz(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dz(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*C-2*B)*z*z*dl_dz(x,y,z,lam))/((c*c+lam)*(c*c+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*C-2*lam*B)*z*z)/((c*c+lam)**3) + (2*(2*lam*C - 2*lam*B)*z)/((c**2 + lam)**2)
            + ((2*A-2*B)*x*x*dl_dz(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*A-2*lam*B)*x*x)/((a*a+lam)**3))
        )
    return v_z

def dw_dx(x,y,z,lam):
    w_x = (g_bold- eta + beta_prime(lam)*S - 2*(alpha(lam) -gamma(lam))*G + x*S*dbetaprime_dx(x,y,z,lam) - 2*x*G*(dalpha_dx(x,y,z,lam) -dgamma_dx(x,y,z,lam)) + y*R*dalphaprime_dx(x,y,z,lam) + 2*y*F*(dgamma_dx(x,y,z,lam) - dbeta_dx(x,y,z,lam))
        + z*V*dbetaprime_dx(x,y,z,lam) - z*U*dalphaprime_dx(x,y,z,lam) - 2*z*C*(dalpha_dx(x,y,z,lam) + dbeta_dx(x,y,z,lam) + dgamma_dx(x,y,z,lam))
        +((-4*z*P(x,y,z,lam)*dP_dx(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (2*z*P2(x,y,z,lam)*dl_dx(x,y,z,lam))/((c**2+lam)**2*Delta(lam)) - (2*z*P2(x,y,z,lam) * dDeltaminus_dx(x,y,z,lam))/(c**2 + lam))
        *((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*A-2*lam*C)*x*x)/((a*a+lam)*(a*a+lam))
        + ((2*lam*B - 2*lam*C)*y*y)/((b*b+lam)*(b*b+lam)))
        - (2*z*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam))
        * ((4*dl_dx(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dx(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dx(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G*z)/((c*c+lam)*(a*a+lam)) + (4*dl_dx(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dx(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dx(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*lam*H*y)/((a**2 + lam)*(b**2 + lam)) + (4*dl_dx(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dx(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dx(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*A-2*C)*x*x*dl_dx(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*A-2*lam*C)*x*x)/((a*a+lam)**3) + (2*(2*lam*A - 2*lam*C)*x)/((a**2 + lam)**2)
            + ((2*B-2*C)*y*y*dl_dx(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dx(x,y,z,lam)*(2*lam*B-2*lam*C)*y*y)/((b*b+lam)**3)
            )
        )
    return w_x

def dw_dy(x,y,z,lam):
    w_y = (x*S*dbetaprime_dy(x,y,z,lam) - 2*x*G*(dalpha_dy(x,y,z,lam) - dgamma_dy(x,y,z,lam)) + f_bold + xii + alpha_prime(lam)*R + 2*F*(gamma(lam)-beta(lam)) + y*R*dalphaprime_dy(x,y,z,lam) + 2*y*F*(dgamma_dy (x,y,z,lam)- dbeta_dy(x,y,z,lam))
        + z*V*dbetaprime_dy(x,y,z,lam)- z*U*dalphaprime_dy(x,y,z,lam) -2*z*C*(dalpha_dy(x,y,z,lam) + dbeta_dy(x,y,z,lam) +dgamma_dy(x,y,z,lam))
                +((-4*z*P(x,y,z,lam)*dP_dy(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (2*z*P2(x,y,z,lam)*dl_dy(x,y,z,lam))/((c**2+lam)**2*Delta(lam)) - (2*z*P2(x,y,z,lam) * dDeltaminus_dy(x,y,z,lam))/(c**2 + lam))
                *((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*A-2*lam*C)*x*x)/((a*a+lam)*(a*a+lam))
                + ((2*lam*B - 2*lam*C)*y*y)/((b*b+lam)*(b*b+lam)))
                - (2*z*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam))
                * ((4*lam*F*z)/((b*b + lam)*(c*c + lam)) + (4*dl_dy(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dy(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dy(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
                    + (4*dl_dy(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dy(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dy(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
                    + (4*lam*H*x)/((a**2 + lam)*(b**2 + lam)) + (4*dl_dy(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dy(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dy(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
                    + ((2*A-2*C)*x*x*dl_dy(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*A-2*lam*C)*x*x)/((a*a+lam)**3)
                    + ((2*B-2*C)*y*y*dl_dy(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dy(x,y,z,lam)*(2*lam*B-2*lam*C)*y*y)/((b*b+lam)**3) + (2*(2*lam*B - 2*lam*C)*y)/((b**2 + lam)**2)
            )
        )
            
    return w_y

def dw_dz(x,y,z,lam):
    w_z = (x*S*dbetaprime_dz(x,y,z,lam) - 2*x*G*(dalpha_dz(x,y,z,lam) - dgamma_dz(x,y,z,lam)) + y*R*dalphaprime_dz(x,y,z,lam) + 2*y*F*(dgamma_dz(x,y,z,lam) - dbeta_dz(x,y,z,lam)) + c_bold + beta_prime(lam)*V -alpha_prime(lam)*U -2*C*(alpha(lam) + beta(lam) + gamma(lam))
        + z*V*dbetaprime_dz(x,y,z,lam) -z*U*dalphaprime_dz(x,y,z,lam) - 2*z*C*(dalpha_dz(x,y,z,lam) + dbeta_dz(x,y,z,lam) +dgamma_dz(x,y,z,lam))
        +((-2*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (-4*z*P(x,y,z,lam)*dP_dz(x,y,z,lam))/((c**2 + lam)*Delta(lam)) + (2*z*P2(x,y,z,lam)*dl_dz(x,y,z,lam))/((c**2+lam)**2*Delta(lam)) - (2*z*P2(x,y,z,lam) * dDeltaminus_dz(x,y,z,lam))/(c**2 + lam))
        *((4*lam*F*y*z)/((b*b+lam)*(c*c+lam)) + (4*lam*G*z*x)/((c*c+lam)*(a*a+lam)) + (4*lam*H*x*y)/((a*a+lam)*(b*b+lam)) + ((2*lam*A-2*lam*C)*x*x)/((a*a+lam)*(a*a+lam))
        + ((2*lam*B - 2*lam*C)*y*y)/((b*b+lam)*(b*b+lam)))
        - (2*z*P2(x,y,z,lam))/((c**2 + lam)*Delta(lam))
        * ((4*lam*F*y)/((b*b+lam)*(c*c+lam)) + (4*dl_dz(x,y,z,lam)*F*y*z)/((b*b+lam)*(c*c+lam)) - (4*lam*F*y*z*dl_dz(x,y,z,lam))/((b*b+lam)**2*(c*c+lam)) - (4*lam*F*y*z*dl_dz(x,y,z,lam))/((b*b+lam)*(c*c+lam)**2)
            + (4*lam*G*x)/((c*c+lam)*(a*a+lam)) + (4*dl_dz(x,y,z,lam)*G*z*x)/((c*c+lam)*(a*a+lam)) - (4*lam*G*z*x*dl_dz(x,y,z,lam))/((c*c+lam)**2*(a*a+lam)) - (4*lam*G*z*x*dl_dz(x,y,z,lam))/((c*c+lam)*(a*a+lam)**2)
            + (4*dl_dz(x,y,z,lam)*H*x*y)/((a*a+lam)*(b*b+lam)) - (4*lam*H*x*y*dl_dz(x,y,z,lam))/((a*a+lam)**2*(b*b+lam)) - (4*lam*H*x*y*dl_dz(x,y,z,lam))/((a*a+lam)*(b*b+lam)**2)
            + ((2*A-2*C)*x*x*dl_dz(x,y,z,lam))/((a*a+lam)*(a*a+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*A-2*lam*C)*x*x)/((a*a+lam)**3) 
            + ((2*B-2*C)*y*y*dl_dz(x,y,z,lam))/((b*b+lam)*(b*b+lam)) - (2*dl_dz(x,y,z,lam)*(2*lam*B-2*lam*C)*y*y)/((b*b+lam)**3)
            )
        )
    return w_z

def pressure(x,y,z,lam):
    p = 2*mu*(2*A*alpha(lam) - (4*A*x**2*P2(x,y,z,lam))/((a**2 + lam)**2 * Delta(lam))  # technically is p_0 + 2mu(...) but set p0 to be 0 as p0 is constant mean pressure at a distance away
                + 2*B*beta(lam) - (4*B*y**2*P2(x,y,z,lam))/((b**2 + lam)**2 * Delta(lam))
                + 2*C*gamma(lam) - (4*C*z**2*P2(x,y,z,lam))/((c**2 + lam)**2 * Delta(lam))
                - (8*F*y*z*P2(x,y,z,lam))/((b**2 + lam)*(c**2 + lam)*Delta(lam))
                - (8*G*x*z*P2(x,y,z,lam))/((a**2 + lam)*(c**2 + lam)*Delta(lam))
                - (8*H*y*x*P2(x,y,z,lam))/((b**2 + lam)*(a**2 + lam)*Delta(lam))
                    )
    return p 


def compute_lambda(x,y,z):
    c1 = 1
    c2 = a**2 + b**2 + c**2 - x**2 - y**2 - z**2
    c3 = a**2*b**2 + a**2*c**2 + b**2*c**2 - x**2*b**2 - x**2*c**2 - y**2*a**2 - y**2*c**2 - z**2*a**2 - z**2*b**2
    c4 = a**2*b**2*c**2 - x**2*b**2*c**2 - y**2*a**2*c**2 - z**2*a**2*b**2
    coeffs = [c1, c2, c3, c4]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    positive_real_root = real_roots[real_roots >= 0][0]
    return positive_real_root
    
def sigma_xx(x,y,z):
    lam = compute_lambda(x,y,z)
    u_x = du_dx(x,y,z,lam)
    p = pressure(x,y,z,lam)
    return -p + 2*u_x

def sigma_xy(x,y,z):
    lam = compute_lambda(x,y,z)
    u_y = du_dy(x,y,z,lam)
    v_x = dv_dx(x,y,z,lam)
    return u_y + v_x

def sigma_xz(x,y,z):
    lam = compute_lambda(x,y,z)
    u_z = du_dz(x,y,z,lam)
    w_x = dw_dx(x,y,z,lam)
    return u_z + w_x

def sigma_yx(x,y,z):
    lam = compute_lambda(x,y,z)
    v_x = dv_dx(x,y,z,lam)
    u_y = du_dy(x,y,z,lam)
    return v_x + u_y

def sigma_yy(x,y,z):
    lam = compute_lambda(x,y,z)
    v_y = dv_dy(x,y,z,lam)
    p = pressure(x,y,z,lam)
    return -p +  2*v_y

def sigma_yz(x,y,z):
    lam = compute_lambda(x,y,z)
    v_z = dv_dz(x,y,z,lam)
    w_y = dw_dy(x,y,z,lam)
    return v_z + w_y

def sigma_zx(x,y,z):
    lam = compute_lambda(x,y,z)
    w_x = dw_dx(x,y,z,lam)
    u_z = du_dz(x,y,z,lam)
    return w_x + u_z

def sigma_zy(x,y,z):
    lam = compute_lambda(x,y,z)
    w_y = dw_dy(x,y,z,lam)
    v_z = dv_dz(x,y,z,lam)
    return w_y + v_z

def sigma_zz(x,y,z):
    lam = compute_lambda(x,y,z)
    w_z = dw_dz(x,y,z,lam)
    p = pressure(x,y,z,lam)
    return -p + 2*w_z

def finite_difference_x(f,x,y,z):
    dx = 1e-6
    dx = 1e-12 * (1 + abs(x) + abs(y) + abs(z))
    plus = f(x+dx,y,z)
    minus = f(x-dx, y,z)
    return (plus - minus)/(2*dx)

def finite_difference_y(f,x,y,z):
    dx = 1e-6
    dx = 1e-12* (1 + abs(x) + abs(y) + abs(z))
    plus = f(x,y+dx,z)
    minus = f(x,y-dx,z)
    return (plus - minus)/(2*dx)

def finite_difference_z(f,x,y,z):
    dx = 1e-6
    dx = 1e-12 * (1 + abs(x) + abs(y) + abs(z))
    plus = f(x,y,z+dx)
    minus = f(x, y,z-dx)
    return (plus - minus)/(2*dx)


# cauchy stress should be symmetric so lets see
print('xy diff', sigma_xy(1,1,1)-sigma_yx(1,1,1))
print('xz diff', sigma_xz(1,1,1)-sigma_zx(1,1,1))
print('yz diff', sigma_yz(1,1,1)-sigma_zy(1,1,1))


eps = 10-(1/np.sqrt(3))

first_comp = finite_difference_x(sigma_xx, 1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps) + finite_difference_y(sigma_xy, 1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps) + finite_difference_z(sigma_xz,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps)
print('first comp', first_comp)

second_comp = finite_difference_x(sigma_yx, 1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps) + finite_difference_y(sigma_yy, 1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps) + finite_difference_z(sigma_yz,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps)
print('second comp', second_comp)

third_comp = finite_difference_x(sigma_zx, 1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps) + finite_difference_y(sigma_zy, 1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps) + finite_difference_z(sigma_zz,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps,1/np.sqrt(3)+eps)
print('third comp', third_comp)

# residuals blow up as we tend towards the surface, i.e as lambda -> 0

eps_values = np.logspace(-5, 5, 100)  # move progressively away from surface
#eps_values = np.linspace(0.01, 2, 500)  # move progressively away from surface
lambda_vals = []

first_vals = []
second_vals = []
third_vals = []

base = 1/np.sqrt(3)

for eps in eps_values:
    x = base + eps
    y = base + eps
    z = base + eps

    lam = compute_lambda(x, y, z)
    lambda_vals.append(lam)

    first_comp = (
        finite_difference_x(sigma_xx, x, y, z)
        + finite_difference_y(sigma_xy, x, y, z)
        + finite_difference_z(sigma_xz, x, y, z)
    )

    second_comp = (
        finite_difference_x(sigma_yx, x, y, z)
        + finite_difference_y(sigma_yy, x, y, z)
        + finite_difference_z(sigma_yz, x, y, z)
    )

    third_comp = (
        finite_difference_x(sigma_zx, x, y, z)
        + finite_difference_y(sigma_zy, x, y, z)
        + finite_difference_z(sigma_zz, x, y, z)
    )

    first_vals.append(first_comp)
    second_vals.append(second_comp)
    third_vals.append(third_comp)

plt.figure()
plt.plot(lambda_vals, first_vals, label="div sigma (x)")
plt.plot(lambda_vals, second_vals, label="div sigma (y)")
plt.plot(lambda_vals, third_vals, label="div sigma (z)")

plt.xlabel("lambda")
plt.ylabel("divergence component")
plt.legend()
# plt.xscale("log")   
# plt.yscale("log")   
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# number of points along the ellipse
n_theta = 200
theta = np.linspace(0, 2*np.pi, n_theta)

# ellipsoid semi-axes
a = 1.0
b = 1.0
c = 1.0
mu = 1.0

# store shear magnitudes
shear_surface = np.zeros_like(theta)

for i, th in enumerate(theta):
    # points on y=0 ellipse slice
    z = a * np.cos(th)
    y = 0.0
    x = c * np.sin(th)
    lam = 0.0  

    gradient = np.array([
        [du_dx(x,y,z,lam), du_dy(x,y,z,lam), du_dz(x,y,z,lam)],
        [dv_dx(x,y,z,lam), dv_dy(x,y,z,lam), dv_dz(x,y,z,lam)],
        [dw_dx(x,y,z,lam), dw_dy(x,y,z,lam), dw_dz(x,y,z,lam)]
    ])

    # Cauchy stress
    sigma = mu * (gradient + gradient.T)

    # outward normal
    P0 = 1/np.sqrt((x*x)/(a**4) + (y*y)/(b**4) + (z*z)/(c**4))
    normal = np.array([P0*x/(a*a), P0*y/(b*b), P0*z/(c*c)])

    # tangential traction
    traction = sigma @ normal
    tangential = traction - np.dot(traction, normal)*normal
    shear_surface[i] = np.linalg.norm(tangential)

# compute arc length along ellipse
dx_dtheta = a * -np.sin(theta)
dz_dtheta = c * np.cos(theta)
ds_dtheta = np.sqrt(dx_dtheta**2 + dz_dtheta**2)
s = np.zeros_like(theta)
s[1:] = np.cumsum(0.5 * (ds_dtheta[1:] + ds_dtheta[:-1]) * np.diff(theta))

# plot
plt.figure(figsize=(7,4))
plt.plot(s, shear_surface, lw=2)
plt.xlabel("Arc length along x-z ellipse slice")
plt.ylabel("Tangential traction magnitude")
plt.title("Tangential traction along particle surface (y=0)")
plt.grid(True)
plt.tight_layout()
plt.show()



