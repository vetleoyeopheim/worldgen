# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:36:26 2020

@author: Vetle
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
GRASS = [34,139,34]
TROPICAL = [15,180,45]
SAVANNAH = [230,230,140]
SAND = [238, 214, 175]
SNOW = [255, 250, 250]
ICE = [228,238,245]
MOUNTAIN = [139, 137, 137]
HIGH_MOUNTAIN = [99,95,95]
TUNDRA = [152,121,101]
DRY_GRASS = [222,232,86]
MARSHLAND = [94,181,126]

COLORS = [DEEP_OCEAN,OCEAN, RIVER, GRASS, TROPICAL, SAVANNAH,\
          SAND, SNOW, ICE, MOUNTAIN, TUNDRA, DRY_GRASS, MARSHLAND]


class Terrain:
    
    def __init__(self, map_height, map_length, scale=128, octaves=7, persistence=0.6, \
                 lacunarity=2.1, water_level=0.41, freeze_pnt=0.09):
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
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
            
        self.n_polar = self.height * 0.1
        self.n_tundra = self.height * 0.15
        self.n_temp = self.height * 0.35
        self.trop = self.height * 0.65
        self.s_temp = self.height * 0.85
        self.s_tundra = self.height * 0.9
        self.s_polar = self.height * 1.0
        
        self.water_level = water_level
        self.d_water_level = self.water_level - 0.1         #Deep water
        self.freeze_pnt = freeze_pnt
        
        self.beach_zone = self.water_level + 0.05
        self.flatland_zone = 0.6
        self.mountain_zone = 1.0
        
        self.seed = random.randint(0,1000)
        
        self.lat_map = self.gen_lat_map()
        self.h_map = self.gen_hmap()
        #self.temp_map = self.gen_temp_map(12)
        self.humid_map = self.gen_humid_map()
        
        self.terr_base = self.gen_terrain_map()         #Basis terrain before ice and snow (temperature effects) are added
        #self.terrain_map = self.apply_temp(12)
    
    def gen_hmap(self):
        noise_map = noisegen.NoiseMap(self.height, self.length)
        h_map = noise_map.gen_perlin_hmap(self.scale, \
                                  self.octaves, self.persistence, self.lacunarity, self.seed)
        return h_map
                
    
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
        temps = self.gen_temp_map(6)
        heights = self.h_map
        humidity = self.humid_map
        terrain_map = np.zeros((self.height, self.length)+(1,))

        #Add extra placeholder dimension to temperature map
        temps = self.dimension_transform(temps)
        heights = self.dimension_transform(heights)
        humidity = self.dimension_transform(humidity)
        
        terrain_map = np.where(heights > 100,OCEAN,GRASS)
        terrain_map = np.where(humidity  < 0.45,DRY_GRASS,terrain_map)

        terrain_map = np.where(heights < self.water_level,OCEAN,terrain_map)
        terrain_map = np.where((heights >= self.water_level) & (heights < self.water_level + 0.04),SAND,terrain_map)
        terrain_map = np.where(heights < self.d_water_level,DEEP_OCEAN,terrain_map)
        terrain_map = np.where((humidity  > 0.53) & (heights > self.water_level + 0.04) & (heights < 0.6),MARSHLAND,terrain_map)
        terrain_map = np.where((humidity  > 0.5) & (temps > 0.5)& (heights > self.water_level + 0.04) & (heights < 0.6),TROPICAL,terrain_map)
        terrain_map = np.where(heights > 0.55,MOUNTAIN,terrain_map)
        terrain_map = np.where(heights > 0.65,HIGH_MOUNTAIN,terrain_map)
        
        return terrain_map
    
    def gen_lat_map(self, symmetric = True):
        """
        Generates a latitude map with values that are smallest towards the pole and smallest close to the equator
        Range of values are 0-1
        Generates a latitude map that is symmetric about the equator by default. If not, distance will be relative to the other pole. That is useful for temperature calculations due to inverse season between south and north
        
        """
        i = np.arange(0,self.height)
        j = np.arange(0,self.length)
        lat_map = np.meshgrid(j,i)
        lat_map = lat_map[1]
        if symmetric is True:
            lat_map = abs(lat_map - self.height/2)
        else:
            lat_map = abs(lat_map - self.height)
        lat_map = (lat_map - lat_map.min())/(lat_map.max() - lat_map.min())     
        lat_map = (lat_map - 1) * (-1)          #This inverts the latitude map so that the equator equals 1 and the poles equals zero
        return lat_map

    def gen_temp_map(self, month):
        """
        Temperature depends on elevation and latitude and month of the year
        Temperature cycle over the year is modeled by a sine function       
        TODO: Implement offset season for northern and southern hemisphere
        """
        lat_map = self.lat_map
        noise_map = (self.h_map - 1) * (-1)

        temp_map = np.multiply(noise_map,lat_map)

        period_coeff = math.pi/365      #Period of the cycle is set to 12
        offset_coeff = 120                #Offset of sine function set to 4 to make the min value equal to 12

        temp_map = temp_map * (0.8 * math.sin(offset_coeff + period_coeff * month) + 0.8)        
        
        #period_coeff = math.pi/6       #Period of the cycle is set to 12
        #offset_coeff = 5                #Offset of sine function set to 4 to make the min value equal to 12

        #temp_map = temp_map * (0.2 * math.sin(offset_coeff + period_coeff * month) + 0.8)
        
        #Add an element of randomness to the temperature
        randarr = np.random.random((self.height, self.length)) * 0.01
        #temp_map = temp_map + randarr
        
        #Increase temperature over the ocean
        temp_map = np.where(temp_map < self.water_level, temp_map + 0.1, temp_map)

        #Apply exponential filter
        temp_map = np.exp(temp_map)
        #Normalize
        temp_map = (temp_map - temp_map.min()) / (temp_map.max() - temp_map.min())
        return temp_map

    def gen_humid_map(self):
        """
        Generates a humidity map based on the height map
        Humidity is inversely related to height so that areas with ocean or close to the ocean gets more humidity
        """
        humid_map = np.zeros((self.height, self.length))
        humid_map = (self.h_map - 1) * (-1)
        #Add an element of randomness to the humidity
        #randarr = np.random.random((self.height, self.length)) + 0.01
        #humid_map = humid_map + randarr
        #Interpolate the humidity map to smooth thins
        humid_map = self.map_interp(humid_map, 3)
        humid_map = (humid_map - humid_map.min())/(humid_map.max() - humid_map.min())

        return humid_map
    
    #slope(gradient) map of the height map
    def get_slope_map(self):
        slope_map = np.gradient(self.h_map)
        
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

    
