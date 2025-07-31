import numpy as np
import scipy as sp
import math
from atmospheric_conditions import Atmosphere


class CF_relations:

    def __init__(self, gama):
        self.gama = gama
    
    # Isentropic Flows Relations
    def IF_To_T(self, M):
        To_T = 1 + (0.5*(gama - 1)* M**2)
        return To_T
    
    def IF_Po_P(self, M):
        Po_P = (1 + (0.5*(gama - 1)* M**2))**(gama/(gama-1))
        return Po_P
    
    def IF_Rhoo_rho(self, M):
        Rhoo_rho = (1 + (0.5*(gama - 1)* M**2))**(1/(gama-1))
        return Rhoo_rho
    
    # Normal Shock Relations - 2 for downstream, 1 for upstream

    # Density ratio across normal shock
    def NS_rho2_rho1(self, M):
        # self.gama = gama
        rho2_rho1 = ((gama+1)*M**2)/(2 + ((gama - 1)*M**2))
        return rho2_rho1
    
    # Pressure ratio across normal shock
    def NS_P2_P1(self, M1):
        P2_P1 = 1 + (2*gama*(M1**2 -1)/(gama+1))
        return P2_P1
    
    # Temperature ratio across normal shock
    def NS_T2_T1(self, M1):
        T2_T1 = (1+(2*gama*(M1**2-1)/(gama+1)))*((2+(gama-1)*M1**2)/((gama+1)*M1**2))
        return T2_T1
    
    # Normal shock downstream Mach number
    def NS_M2(self, M1):
        # self.M1 = M
        # self.gama = gama
        M2 = np.sqrt((1+((gama-1)*M1**2 /2))/((gama*M1**2)-((gama-1)/2)))
        return M2

    def NS_Po2_Po1(self, M1):
        M2 = self.NS_M2(M1)
        Po1_P1 = self.IF_Po_P(M1)
        Po2_P2 = self.IF_Po_P(M2)
        P2_P1 = self.NS_P2_P1(M1)
        Po2_Po1 = Po2_P2/Po1_P1 * P2_P1

        return Po2_Po1
    
    # Oblique Shock (OS) relations

    # OS: Mach number normal component # beta is in radians
    def OS_M1n(self, beta, M1):
        beta_rad = beta * np.pi/180
        M1n = M1 * np.sin(beta_rad)
        return M1n
    
    # OS: normal component of downstream Mach number # beta is in deg for the argument
    def OS_M2n(self, beta, M1):
        # beta_rad = beta * np.pi/180 # conversion to rad from deg for calculation
        M1n = self.OS_M1n(beta, M1)
        M2n = self.NS_M2(M1n)
        return M2n

    # OS: Downstream Mach number
    def OS_M2(self, M1, beta, theta):
        beta_rad = beta * np.pi/180
        theta_rad = theta* np.pi / 180
        M2n = self.OS_M2n(beta, M1)
        M2 = M2n/np.sin(beta_rad-theta_rad)
        return M2

    # Oblique shock angle
    def calc_residual(self, M1, theta, beta: np.float64):
        theta_rad = np.pi * theta/180 # radian
        residual = np.tan(theta_rad) - 2.0 * (1.0/np.tan(beta)) * ( M1**2.0 * (np.sin(beta))**2.0 - 1.0 )/( M1**2.0 * (gama + np.cos(2*beta)) + 2.0 )
        return residual
    
    def calc_jacobian(self, M, beta: np.float64):
        F = ( M**2.0 * (np.sin(beta))**2.0 - 1.0 )/( M**2.0 * (gama + np.cos(2*beta)) + 2.0 )
        dFdb = (M**2.0 * np.sin(2*beta))/( M**2.0 * (gama + np.cos(2*beta)) + 2.0 ) +\
            ( (M**2.0 * (np.sin(beta))**2.0 - 1.0) * (2.0 * M**2.0 * np.sin(2*beta)) )/( M**2.0 * (gama + np.cos(2*beta)) + 2.0 )**2.0
        return -2.0 * ( F/(np.sin(beta))**2.0 + (1.0/np.tan(beta)) * dFdb )
    
    def OS_beta(self, theta, M):
        # theta_rad = theta_deg * np.pi/180
        beta0 = np.pi/5.0
        calc_residual = lambda x: self.calc_residual(M, theta, x)
        calc_jacobian = lambda x: self.calc_jacobian(M, x)
        b  = sp.optimize.fsolve(calc_residual, beta0, fprime=calc_jacobian, xtol=1.0e-12)
        beta = b[0] * 180/np.pi # converting rad to deg
        return beta
    
    def OS_P2_P1(self, M1n):
        return self.NS_P2_P1(M1n)
    
    def OS_T2_T1(self, M1n):
        return self.NS_T2_T1(M1n)
    
    def OS_rho2_rho1(self, M1n):
        return self.NS_rho2_rho1(M1n)
    
    def OS_Po2_Po1(self, M1n):
        return self.NS_Po2_Po1(M1n)
    

    # Expansion Wave (EW) relations

    def EW_nu1(self, M):
        nu = np.sqrt( (gama+1.0)/(gama-1.0) ) * np.arctan( np.sqrt( (M**2.0 - 1.0) * (gama-1.0)/(gama+1.0) ) )-\
        np.arctan( np.sqrt( M**2.0 - 1.0 ) )
        return nu * 180/np.pi # converting nu to deg from rad
    
    def EW_nu_residual(self, nu, M: np.float64):
        residual = (np.sqrt((gama+1)/(gama-1))*np.arctan(np.sqrt((gama-1)/(gama+1)*(M**2-1)))-\
                    np.arctan(np.sqrt(M**2-1))-nu)
        return residual

    def EW_nu_jacobian(self, M: np.float64):
        jacobian = -1/(M * np.sqrt(-1 + M**2)) + (M * (-1 + gama) * np.sqrt((1 + gama)/(-1 + gama)))/\
            (np.sqrt(((-1 + M**2)*(-1 + gama))/(1 + gama))*(1 + gama)*(1 + ((-1 + M**2)*(-1 + gama))/(1 + gama)))
        return jacobian
    
    # Calculating M2 from nu using Newton Raphson
    def EW_M2(self, nu):
        nu_rad = nu * np.pi/180
        M0 = 2 # Initial guess of downstream Mach number
        residual = lambda x: self.EW_nu_residual(nu_rad, x)
        jacobian = lambda x: self.EW_nu_jacobian(x)
        M2 = sp.optimize.fsolve(residual, M0, fprime=jacobian, xtol=1.0e-12)
        return M2[0]
    
    # Expansion wave Temperature ratios
    def EW_T1_T2(self, M1, M2):
       return (1+(gama-1)*M2**2 /2)/(1+(gama-1)*M1**2 /2)
    
    def EW_P1_P2(self, M1, M2):
        return self.EW_T1_T2(M1, M2)**(gama/(gama-1))
    
    def EW_rho1_rho2(self, M1, M2):
        return self.EW_T1_T2(M1, M2)**(1/(gama-1))
    
    # Isentropic Converging-Diverging nozzle
    def CD_residual(self,A_As, M: np.float64):
        residual = (1/M**2*(2/(gama +1)*(1+ (gama-1)/2*M**2))**((gama + 1)/(gama - 1))) - A_As**2
        return residual
    
    def CD_jacobian(self, M: np.float64):
        jacobian = -(2**(1 + (1 + gama)/(-1 + gama)) * ((1 + 1/2 * M**2 *(-1 + gama))/(1 + gama))**((1 + gama)/(-1 + gama)))/M**3 +\
            (2**((1 + gama)/(-1 + gama))* ((1 + 1/2 * M**2 *(-1 + gama))/(1 + gama))**(-1 + (1 + gama)/(-1 + gama)))/M
        return jacobian
    
    # CD Mach number from A/A*
    def CD_M(self, A_As, M0):   # M0 is the initial guess for the Mach number
        residual = lambda x: self.CD_residual(A_As, x)
        jacobian = lambda x: self.CD_jacobian(x)
        M = sp.optimize.fsolve(residual, M0, fprime=jacobian, xtol=1.0e-12)
        return M[0] 
    
    # CD A/A* from M
    def CD_A_As(self, M):
        A_As = np.sqrt(1/M**2 *(2/(gama+1)*(1+(gama-1)*0.5*M**2))**((gama+1)/(gama-1)))
        return A_As
    
    


# if __name__ == "__main__":
def Concorde_calculate(M1, gama,theta1, theta2, theta3, A_As, altitude):
    # M1 = 2.0
    # gama = 1.4 # air
    # theta1 = 10 # (deg) - inlet wedge turn angle
    # theta2 = 35 # (deg) - ramp 1 deflection angle
    # theta3 = 40 # deg - Expansion angle
    # A_As = 1.3
    # altitude = 18300
    
    
    atmosphere = Atmosphere()
    Concorde = CF_relations(gama)

    # Atmospheric conditions at the flight altitude [Region 1]
    T1 = atmosphere.get_temperature(altitude)
    P1 = atmosphere.get_pressure(altitude)
    R1 = atmosphere.get_density(altitude)

    To1 = Concorde.IF_To_T(M1)*T1
    Po1 = Concorde.IF_Po_P(M1)*P1
    Ro1 = Concorde.IF_Rhoo_rho(M1)*R1 
    
    print("==================================================================================")
    print(f"Atmospheric Conditions at flight altitude({altitude}m) [Region 1]")
    print(f"M1: {M1}")
    print(f"T1 = {T1} K")
    print(f"P1 = {P1} Pa")
    print(f"Rho1 = {R1} kg/m^3")
    print(f"To1 = {To1} K")
    print(f"Po1 = {Po1} Pa")
    print(f"Rhoo1 = {Ro1} kg/m^3")
    print("==================================================================================")


    # Flow across the first oblique shock [Region 2]
    beta = Concorde.OS_beta(theta1, M1)
    M1n = Concorde.OS_M1n(beta, M1)
    # M2n = Concorde.OS_M2n(beta, M1)
    M2 = Concorde.OS_M2(M1, beta, theta1)
    P2 = Concorde.OS_P2_P1(M1n)*P1
    T2 = Concorde.OS_T2_T1(M1n)*T1
    R2 = Concorde.OS_rho2_rho1(M1n)*R1
    To2 = Concorde.IF_To_T(M2)*T2
    Po2 = Concorde.IF_Po_P(M2)*P2
    Ro2 = Concorde.IF_Rhoo_rho(M2)*R2
    
    print("==================================================================================")
    print(f"Flow properties across the first oblique shock [Region 2]")
    print(f"beta = {beta} deg")
    print(f"M2 = {M2}")
    print(f"T2 = {T2} K")
    print(f"P2 = {P2} Pa")
    print(f"Rho2 = {R2} kg/m^3")
    print(f"To2 = {To2} K")
    print(f"Po2 = {Po2} Pa")
    print(f"Rhoo2 = {Ro2} kg/m^3")
    print("==================================================================================")
    
    # Flow across the second oblique shock [Region 3]
    theta_2 = theta2 - theta1
    beta2 = Concorde.OS_beta(theta_2, M2)
    M2n = Concorde.OS_M1n(beta2, M2)
    # M3n = Concorde.OS_M2n(beta2, M2)
    M3 = Concorde.OS_M2(M2, beta2, theta_2)
    P3 = Concorde.OS_P2_P1(M2n)*P2
    T3 = Concorde.OS_T2_T1(M2n)*T2
    R3 = Concorde.OS_rho2_rho1(M2n)*R2
    To3 = Concorde.IF_To_T(M3)*T3
    Po3 = Concorde.IF_Po_P(M3)*P3
    Ro3 = Concorde.IF_Rhoo_rho(M3)*R3
    
    print("==================================================================================")
    print(f"Flow properties across the second oblique shock [Region 3]")
    print(f"theta 2 = {theta_2} deg")
    print(f"beta 2 = {beta2} deg")
    print(f"M3 = {M3}")
    print(f"T3 = {T3} K")
    print(f"P3 = {P3} Pa")
    print(f"Rho3 = {R3} kg/m^3")
    print(f"To3 = {To3} K")
    print(f"Po3 = {Po3} Pa")
    print(f"Rhoo3 = {Ro3} kg/m^3")
    print("==================================================================================")

    # Flow properties across the expansion wave [Region 4]
    nu = Concorde.EW_nu1(M3)
    theta3p = theta3 - theta2
    assert theta3 >= theta2, "there is no expansion based on the upstream flow angle (theta 2 >= theta 3)"

    nu2 = theta3p + nu
    M4 = Concorde.EW_M2(nu2)
    P4 = P3/Concorde.EW_P1_P2(M3, M4)
    T4 = T3/Concorde.EW_T1_T2(M3, M4)
    R4 = R3/Concorde.EW_rho1_rho2(M3, M4)
    To4 = To3
    Po4 = Po3
    # Ro4 = Ro3
    Ro4 = Concorde.IF_Rhoo_rho(M4)*R4

    print("==================================================================================")
    print(f"Flow properties across the expansion wave [Region 4]")
    print(f"nu = {nu} deg")
    print(f"nu 2 = {nu2} deg")
    print(f"M4 = {M4}")
    print(f"T4 = {T4} K")
    print(f"P4 = {P4} Pa")
    print(f"Rho4 = {R4} kg/m^3")
    print(f"To4 = {To4} K")
    print(f"Po4 = {Po4} Pa")
    print(f"Rhoo4 = {Ro4} kg/m^3")
    print("==================================================================================")

    # Flow properties across normal shock [Region 5]
    M5 = Concorde.NS_M2(M4)
    P5 = Concorde.NS_P2_P1(M4)*P4
    T5 = Concorde.NS_T2_T1(M4)*T4
    R5 = Concorde.NS_rho2_rho1(M4)*R4
    Po5 = Concorde.NS_Po2_Po1(M4)*Po4
    To5 = Concorde.IF_To_T(M5)*T5
    Ro5 = Concorde.IF_Rhoo_rho(M5)*R5

    print("==================================================================================")
    print(f"Flow properties across normal shock wave [Region 5]")
    print(f"M5 = {M5}")
    print(f"T5 = {T5} K")
    print(f"P5 = {P5} Pa")
    print(f"Rho5 = {R5} kg/m^3")
    print(f"To5 = {To5} K")
    print(f"Po5 = {Po5} Pa")
    print(f"Rhoo5 = {Ro5} kg/m^3")
    print("==================================================================================")

    # Isentropic expansion in the diverging nozzle [Region 6]
    M0 = 0.5
    M6 = Concorde.CD_M(A_As, M0)
    Po6 = Po5
    To6 = To5
    Ro6 = Ro5
    P6 = Po6/Concorde.IF_Po_P(M6)
    T6 = To6/Concorde.IF_To_T(M6)
    R6 = Ro6/Concorde.IF_Rhoo_rho(M6)

    print("==================================================================================")
    print(f"Isentropic expansion in the diverging nozzle [Region 6]")
    print(f"M6 = {M6}")
    print(f"T6 = {T6} K")
    print(f"P6 = {P6} Pa")
    print(f"Rho6 = {R6} kg/m^3")
    print(f"To6 = {To6} K")
    print(f"Po6 = {Po6} Pa")
    print(f"Rhoo6 = {Ro6} kg/m^3")
    print("==================================================================================")

    #################################### TESTING ###################################
    # Parameters
    # M = 3.0
    # gama = 1.4 # air
    # theta_deg = 20 #deg
    # theta_deg_2 = 5 #deg - Expansion angle
    # Concorde = CF_relations(gama)

    # Oblique Shock
    # beta = Concorde.OS_beta(theta_deg, M)

    # print(f"beta(deg)= {beta}")
    # M1n = Concorde.OS_M1n(beta, M)
    # M2n = Concorde.OS_M2n(beta, M)
    # M2 = Concorde.OS_M2(M, beta, theta_deg)

    # P2P1 = Concorde.OS_P2_P1(M1n)
    # T2T1 = Concorde.OS_T2_T1(M1n)
    # r2r1 = Concorde.OS_rho2_rho1(M1n)
    # Po2Po1 = Concorde.OS_Po2_Po1(M1n)

    # #Expansion wave
    # nu_1 = Concorde.EW_nu1(M)
    # theta2 = theta_deg_2
    # nu_2 = theta2 + nu_1

    # EW_M2 = Concorde.EW_M2(nu_2)
    # print(f"nu_2 = {nu_2}")
    # print(f"EW M2 = {EW_M2}")
    # print(f"nu1 = {nu_1}")

    # EW_T1T2 = Concorde.EW_T1_T2(M, EW_M2)
    # EW_P1P2 = Concorde.EW_P1_P2(M, EW_M2)
    # EW_R1R2 = Concorde.EW_rho1_rho2(M, EW_M2)

    # print(f"EW T1/T2 = {EW_T1T2}")
    # print(f"EW_P1/P2 = {EW_P1P2}")
    # print(f"EW rho1/rho2 = {EW_R1R2}")

    # print("")
    # print("CD Nozzle")
    # CD_M2 = Concorde.CD_M(2, 0.9)
    # CD_A_As = Concorde.CD_A_As(3)

    # print(f"CD M2 = {CD_M2}")
    # print(f"CD A/A* = {CD_A_As}")

    # print("")
    # print("Oblique shock tests")
    # print(f"M2n = {M2n}")
    # print(f"M2 = {M2}")
    # print(f"OS P2/P1 = {P2P1}")
    # print(f"OS T2/T1 = {T2T1}")
    # print(f"OS rho2/rho1 = {r2r1}")
    # print(f"OS Po2/Po1 = {Po2Po1}")

    
    # Isentropic Flow tests
    # test = 1/Concorde.IF_To_T(M)
    # test = 1/Concorde.IF_Po_P(M)
    # test = 1/Concorde.IF_Rhoo_rho(M)


    # NS Tests 

    # test1 = Concorde.NS_P2_P1(M)
    # test2 = Concorde.NS_rho2_rho1(M)
    # test3 = Concorde.NS_T2_T1(M)
    # test4 = Concorde.NS_M2(M)
    # test5 = Concorde.NS_Po2_Po1(M)
    # print(test)
    
    # print(" ")
    # print("Normal shock tests")
    # print(f"P2/P1 = {test1}")
    # print(f"rho2/rho1 = {test2}")
    # print(f"T2/T1 = {test3}")
    # print(f"M2 = {test4}")
    # print(f"Po2/Po1 = {test5}")


if __name__ =="__main__":
    M1 = 3
    gama = 1.4 # air
    theta1 = 10 # (deg) - inlet wedge turn angle

    # theta2 ranges from 15 - 35 in this code but effectively 5 to 25 
    # (i.e 15 deg is effectively 5 and 35 is effectively 25)
    theta2 = 35 # (deg) - ramp 1 deflection angle
    theta3 = 40 # deg - Expansion angle
    A_As = 1.3
    altitude = 18300
    Concorde_calculate(M1, gama, theta1, theta2, theta3, A_As, altitude)