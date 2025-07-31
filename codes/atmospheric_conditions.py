""" 
This script is used to estimate the atmospheric properties from sea level
up to an altitude of 30,000 meters  above sea level
"""
import numpy as np
class Atmosphere:

    def get_region(self, altitude):
        # self.altitude = altitude

        if altitude <= 11000:
            region = 'Troposphere'
        elif 11000 < altitude <= 20000:
            region = 'Tropopause'
        elif 20000 < altitude <= 30000:
            region = 'Stratosphere'
        
        return region

    def get_temperature(self, altitude):
        self.altitude = altitude
        
        if self.altitude <= 11000:
            temperature = 288.15 -(0.0065*altitude)
        elif  11000 < self.altitude <= 20000:
            temperature = 216.65
        elif 20000 < self.altitude <= 30000:
            temperature = 216.65 + (0.001 * (altitude - 20000))
        return temperature
    
    def get_density(self, altitude):
        self.altitude = altitude

        if self.altitude <= 11000:
            relative_density = (1-(0.0065*altitude/288.15))**4.25587971
        elif  11000 < self.altitude <= 20000:
            relative_density = 0.297076/np.exp(0.000157689*altitude-1.734579)
        elif 20000 < self.altitude <= 30000:
            temperature = 216.65 + (0.001 * (altitude - 20000))
            relative_density = 15.569627/(temperature*(temperature/216.65)**34.1632)

        return relative_density * 1.225

    def get_pressure(self, altitude):
        self.altitude = altitude

        relative_density = self.get_density(altitude)/1.225
        relative_temperature = self.get_temperature(altitude)/288.15
        relative_pressure = relative_density * relative_temperature

        return relative_pressure * 101325
    
    def get_dynamic_viscosity(self, altitude):
        self.altitude = altitude
        temp = self.get_temperature(altitude)
        temp_sl = 288.15 # sea level; temperature
        temp_s = 110 # Sutherland constant Ts = 110K
        self.relative_viscosity  =((temp/temp_sl)**(3/2))*((temp_sl+temp_s)/(temp + temp_s))

        return self.relative_viscosity * 1.7894e-5


if __name__ == "__main__":


    height = 5000
    assert(height <= 30000), "maximum height/altitude to use this model to predict the atmospheric condition is 30,000 meters"
    my_Atmos = Atmosphere()
    temperature = my_Atmos.get_temperature(height)
    density = my_Atmos.get_density(height)
    pressure = my_Atmos.get_pressure(height)
    viscosity = my_Atmos.get_dynamic_viscosity(height)
    print(f"The region is {my_Atmos.get_region(height)}")
    print(f"Temperature = {temperature}")
    print(f"Density = {density}")
    print(f"Pressure = {pressure}")
    print(f"relative temperature = {temperature/288.15}")
    print(f"relative pressure = {pressure/101325}")
    print(f"relative density = {density/1.225}")
    print(f"relative viscosity = {viscosity}")

    print(my_Atmos.get_dynamic_viscosity(height))
