import numpy as np
import scipy as sp

def calc_residual(b: np.float64):
    theta = np.pi/6.0
    M = 3.0
    Gamma = 1.4
    return np.tan(theta) - 2.0 * (1.0/np.tan(b)) * ( M**2.0 * (np.sin(b))**2.0 - 1.0 )/( M**2.0 * (Gamma + np.cos(2*b)) + 2.0 )

def calc_jacobian(b: np.float64):
    # theta = np.pi/6.0
    M = 3.0
    Gamma = 1.4
    F = ( M**2.0 * (np.sin(b))**2.0 - 1.0 )/( M**2.0 * (Gamma + np.cos(2*b)) + 2.0 )
    dFdb = (M**2.0 * np.sin(2*b))/( M**2.0 * (Gamma + np.cos(2*b)) + 2.0 ) +\
        ( (M**2.0 * (np.sin(b))**2.0 - 1.0) * (2.0 * M**2.0 * np.sin(2*b)) )/( M**2.0 * (Gamma + np.cos(2*b)) + 2.0 )**2.0
    return -2.0 * ( F/(np.sin(b))**2.0 + (1.0/np.tan(b)) * dFdb )

if __name__ == "__main__":
    b0 = np.pi/6.0
    # f  = lambda x: calc_residual(x)
    # df = lambda x: calc_jacobian(x)
    # b  = sp.optimize.fsolve(f, b0, fprime=df)
    b  = sp.optimize.fsolve(calc_residual, b0, fprime=calc_jacobian, xtol=1.0e-12)
    
    M = 3.0
    Gamma= 1.4
    nu = np.sqrt( (Gamma+1.0)/(Gamma-1.0) ) * np.arctan( np.sqrt( (M**2.0 - 1.0) * (Gamma-1.0)/(Gamma+1.0) ) )-\
        np.arctan( np.sqrt( M**2.0 - 1.0 ) ) 
    
    print("beta(deg)= ", b*180.0/np.pi)
    print("nu= ", nu*180.0/np.pi)