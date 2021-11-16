"""
@author: Vetle Øye Opheim
"""


import noise
from PIL import Image
import random
import math
import numpy as np
import noisegen
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from numpy import inf

BLANK = [0,0,0]
DEEP_OCEAN = [30,75,235]
OCEAN = [65,105,225]
RIVER = [65,105,240]
FORREST = [34,139,34]
TROPICAL = [15,180,45]
SAVANNAH = [230,230,140]
SAND = [238, 214, 175]
SNOW = [255, 250, 250]
ICE = [228,238,245]
MOUNTAIN = [139, 137, 137]
HIGH_MOUNTAIN = [99,95,95]
TUNDRA = [152,121,101]
BORREAL_FORREST = [33,82,16]
DRY_GRASS = [222,232,86]
MARSHLAND = [94,181,126]

COLORS = [DEEP_OCEAN,OCEAN, RIVER, FORREST, TROPICAL, SAVANNAH,\
          SAND, SNOW, ICE, MOUNTAIN, TUNDRA, DRY_GRASS, MARSHLAND,\
          BORREAL_FORREST]


class Terrain:
    
    def __init__(self, map_height, map_length, octaves=7, persistence=0.6, \
                 lacunarity=2.1, water_level=0.41, freeze_pnt=0.15):
        """

        Parameters
        ----------
        map_height : 
            Number of pixels of the map in the y direction
        map_length : 
            Number of pixels of the map in the y direction
        scale :  optional
            parameter for noise generation. The default is 128.
        octaves :  optional
            parameter for noise generation. The default is 6.
        persistence :  optional
            parameter for noise generation. The default is 0.6.
        lacunarity :  optional
            parameter for noise generation. The default is 2.0.
        water_level :  optional
            determine the height of the oceans, lakes etc., must be between 0 and 1. The default is 0.4.
        freeze_pnt :  optional
            Determine how cold the world should be. Value between 0 and 1. The default is 0.35.

        """
        self.height = map_height
        self.length = map_length
        self.scale = self.length / 4
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
            
        self.water_level = water_level
        self.d_water_level = self.water_level - 0.1         #Deep water
        self.freeze_pnt = freeze_pnt
        
        self.beach_zone = self.water_level + 0.05
        
        self.seed = random.randint(0,1000)

        self.h_map = self.gen_hmap()
        self.surf_hmap = self.surface_hmap()
        self.temp_map = self.gen_temp_map(9)    #The standard temperature map
        self.humid_map = self.gen_humid_map()
        
        self.terr_base = self.gen_terrain_map()         #Basis terrain before ice and snow (temperature effects) are added
        #self.terrain_map = self.apply_temp(12)
    
    def gen_hmap(self):
        noise_map = noisegen.NoiseMap(self.height, self.length)
        h_map = noise_map.gen_simplex_map(128, 7, 0.6, 2.1, self.seed)
        return h_map

    def surface_hmap(self):
        """
        The surface height map is the regular height map but with heights over water equal to the water level
        This ensures that the oceans are completely flat, and can thereby be used for calculations of wind flow
        """
        surf_hmap = np.where(self.h_map <= self.water_level, self.water_level, self.h_map)
        return surf_hmap
                
    
    #Assign colors to map based on height_map and longitude()
    def gen_terrain_map(self):
        t_map = self.set_terrain_type()
        return t_map
    
    #Turn a 2d array of floats into a 3d array to be compatible for masking
    def dimension_transform(self, amap):
        amap = np.reshape(amap, amap.shape +(1,))
        return amap

    #Apply temp map to terrain map to determine which
    def apply_temp(self, month):
        terrain_map = self.terr_base
        temps = self.gen_temp_map(month)
        temps = self.dimension_transform(temps)
        heights = self.h_map
        heights = self.dimension_transform(heights)
        terrain_map = np.where((temps < self.freeze_pnt) & (heights > self.water_level),SNOW,terrain_map)
        terrain_map = np.where((temps < self.freeze_pnt) & (heights <= self.water_level),ICE,terrain_map)
        return terrain_map

    def set_terrain_type(self):
        temps = self.gen_temp_map(9)        #Temperature in september as the "average" temperature of the year is what will determine the biome
        heights = self.h_map
        humidity = self.humid_map
        terrain_map = np.zeros((self.height, self.length)+(1,))

        #Add extra placeholder dimension to temperature map
        temps = self.dimension_transform(temps)
        heights = self.dimension_transform(heights)
        humidity = self.dimension_transform(humidity)

        terrain_map = np.where(heights > 100,OCEAN,FORREST)
        #Determine tundra zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps < 0.2),TUNDRA,terrain_map)
        #Determine borreal forrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.2),BORREAL_FORREST,terrain_map)
        #Determine grassland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity < 0.3),DRY_GRASS,terrain_map)
        #Determine temperate forrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.4) & (humidity < 0.4),FORREST,terrain_map)
        #Determine marshland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps < 0.45) & (humidity > 0.5),MARSHLAND,terrain_map)
        #Determine rainforrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.45) & (humidity > 0.35),TROPICAL,terrain_map)
        #Determine savannah zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.45) & (humidity > 0.35),SAVANNAH,terrain_map)
        #Determine desert zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.45) & (humidity < 0.25),SAND,terrain_map)
        #Determine mountain zone
        terrain_map = np.where((heights > 0.6),MOUNTAIN,terrain_map)
        #Determine high mountain zone
        terrain_map = np.where((heights > 0.75),HIGH_MOUNTAIN,terrain_map)
        #Determine beach  zone
        terrain_map = np.where((heights > self.water_level) & (heights <= self.beach_zone),SAND,terrain_map)
        #Determine ocean  zone
        terrain_map = np.where((heights > self.d_water_level) & (heights <= self.water_level),OCEAN,terrain_map)
        #Determine deep ocean  zone
        terrain_map = np.where((heights < self.d_water_level),DEEP_OCEAN,terrain_map)
        
        return terrain_map
    

    def gen_temp_map(self, month):
        """
        Temperature depends on elevation and latitude and month of the year
        Temperature cycle over the year is modeled by a sine function       
        TODO: Implement offset season for northern and southern hemisphere
        """
        temp_map = (self.h_map - 1) * (-1)
        lat_map = noisegen.LatMap(self.height, self.length)
        latitude = lat_map.gen_lat_map()
        
        temp_map = (temp_map * latitude)

        #Some lines for daily simulation for a year that has been commented out
        #period_coeff = math.pi/365      #Period of the cycle is set to 12
        #offset_coeff = 120                #Offset of sine function set to 4 to make the min value equal to 12
        #temp_map = temp_map * (0.8 * math.sin(offset_coeff + period_coeff * month) + 0.8)        
        
        #Make the temperature cycle with the input month
        period_coeff = math.pi/6       #Period of the cycle is set to 12
        offset_coeff = 5                #Offset of sine function set to 4 to make the min value equal to 12
        #temp_map = temp_map * (2 * math.sin(offset_coeff + period_coeff * month) + 0.8)
        
        #Increase temperature over the ocean
        temp_map = np.where(self.h_map < self.water_level, temp_map + 0.05, temp_map)
        #Normalize
        temp_map = (temp_map - temp_map.min()) / (temp_map.max() - temp_map.min())
        return temp_map

    def gen_humid_map(self):
        """
        Generates a humidity map based on the height map
        Humidity is inversely related to height so that areas with ocean or close to the ocean gets more humidity
        """
        humid_map = np.exp((self.h_map - 1) * (-1))

        #Normalize
        humid_map = (humid_map - humid_map.min())/(humid_map.max() - humid_map.min())

        return humid_map
    
    #slope(gradient) map of a height map
    def get_slope_map(self, amap):
        slope_map = np.gradient(amap)
        
        return slope_map

    def map_interp(self, amap, iterations):
        interp_map = amap
        for iter in range(iterations):
            i = np.arange(0,self.height)
            j = np.arange(0,self.length)
            #interp = interp2d(i,j,amap)
            #interp_map = interp(i,j)
            interp = RectBivariateSpline(i,j,amap)
            interp_map = interp(i,j)
        
        return interp_map


    def wind_flow(self):
        """
        STILL EXPERIMENTAL
        Calculate a wind flow map based on a 2D Burgers equation
        Calculations are based on the height map. A negative slope accelerates wind speed due to gravity (g). Opposite is true for upward facing slopes
        Uses conservation of mass principle
        Equations are:
            dv_dx = -du_dy - g * s_x(x,y)
            du_dy = -dv_dx - g * s_y(x,y)
        """
        latmap = noisegen.LatMap(self.height, self.length)
        surface = self.surf_hmap
        slopes = self.get_slope_map(surface)
        latitude = latmap.gen_lat_map() * (-1)
        T = 6
        t = 1
        g = 9.8
        v = 0.25
        h = 0.1

        wind_x = np.zeros((T, self.height, self.length))
        wind_x[0] = 1
        wind_y = np.zeros((T, self.height, self.length))
        wind_y[0] = 0

        while t < T:
            i = 0
            v_arr = np.zeros((self.height, self.length))
            u_arr = np.zeros((self.height, self.length))
            temp_map = self.gen_temp_map(t)     #Generate the temperature map for the month
            temp_gradient = self.get_slope_map(temp_map)       #Calculate the temperature gradient
            while i < self.height - 1:
                j = 0
                v_row = np.zeros(self.length)
                u_row = np.zeros(self.length)
                while j < self.length - 1:
                    if j == 0:
                        vx = 0
                        vy = 0
                    elif j == self.length:
                        vx = 0
                        vy = 0
                    elif i == 0:
                        vx = 0
                        vy = 0
                    elif i == self.height:
                        vx = 0
                        vy = 0
                    else:             
                        vx = wind_x[t - 1][i][j]\
                           - wind_x[t - 1][i][j] * (wind_x[t - 1][i][j] - wind_x[t - 1][i - 1][j])\
                           - wind_y[t - 1][i][j] * (wind_x[t - 1][i][j] - wind_x[t - 1][i][j - 1])\
                           - g * slopes[0][i][j] - h * temp_gradient[0][i][j] + h * temp_map[i][j]

                        vy = wind_y[t - 1][i][j]\
                           - wind_y[t - 1][i][j] * (wind_y[t - 1][i][j] - wind_y[t - 1][i][j - 1])\
                           - wind_x[t - 1][i][j] * (wind_y[t - 1][i][j] - wind_y[t - 1][i - 1][j])\
                           - g * slopes[1][i][j] - h * temp_gradient[1][i][j] + h * temp_map[i][j]
                    v_row[j] = vx
                    u_row[j] = vy
                    j += 1
                v_arr[i] = v_row
                u_arr[i] = u_row
                i += 1
            wind_x[t] = v_arr
            wind_y[t] = u_arr
            t += 1
        return wind_x, wind_y

class Climate:

    def __init__(self, h_map):
        pass
