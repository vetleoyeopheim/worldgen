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

from numpy import inf




MAP_HEIGHT = 256   
MAP_LENGTH = 512
MAX_ELEVATION = 1
PEAK_NUM = 12

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
TUNDRA = [152,121,101]
DRY_GRASS = [222,232,86]
MARSHLAND = [94,181,126]

COLORS = [DEEP_OCEAN,OCEAN, RIVER, GRASS, TROPICAL, SAVANNAH,\
          SAND, SNOW, ICE, MOUNTAIN, TUNDRA, DRY_GRASS, MARSHLAND]


SAVE_PATH = r"C:\Users\Vetle\Documents\Python\worldgen\\"

### "i" refers to global height coordinates of the map (latitude) and is equivalent to y coordinates
### "j" refers to global length coordinates of the map (longitude) and is equivalent to y coordinates

class World:
    
    def __init__(self, map_height, map_length, scale=128, octaves=6, persistence=0.6, \
                 lacunarity=2.0, water_level=0.4, freeze_pnt=0.35):
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
        self.freeze_pnt = freeze_pnt
        
        self.beach_zone = self.water_level + 0.05
        self.flatland_zone = 0.6
        self.mountain_zone = 1.0
        
        self.seed = random.randint(0,100)
        
        self.h_map = self.gen_hmap()
        self.temp_map = self.gen_temp_map(6)
        self.humid_map = self.gen_humid_map()
        self.river_sources = self.gen_river_sources()
        
        self.terr_map = self.gen_terrain_map()
    
    
    def gen_hmap(self):

        h_map = noisegen.gen_perlin_hmap(self.height, self.length, 32, \
                                         self.octaves, self.persistence, self.lacunarity, self.seed)
        
        h_map = self.exp_filter(h_map)
        
        h_map = self.normalize_array(h_map)
        #h_map = self.circular_gradient_transform(h_map)
        return h_map
    
    #Exponential filter for noise function
    def exp_filter(self,h_map):
        h_map = np.exp(h_map)
        return h_map
                
    
    def normalize_array(self, array):
        array_norm = (array - array.min())/(array.max() - array.min())
        return array_norm
    
    #Get an array with all coordinates above land(excludes oceans)
    def get_land_coords(self, h_map):
        land_area = []
        for i in range(self.height):
            for j in range(self.length):
                
                elevation = h_map[i][j]
                if elevation > self.water_level:
                    land_area.append([i,j])
        return land_area
                
    
    #Assign colors to map based on height_map and longitude()
    def gen_terrain_map(self):
        t_map = np.zeros((self.height, self.length)+(3,))
        for i in range(self.height):
            for j in range(self.length):
                cell = self.set_terrain_type(i, j)
                t_map[i][j] = cell

        return t_map
    
    def get_climate_zone(self, i):
        latitude = i
        
        if latitude < self.n_polar:
            return ("n_polar")
        elif latitude < self.n_tundra:
            return ("n_tundra")
        elif latitude < self.n_temp:
            return ("n_temp")
        elif latitude < self.trop:
            return ("trop")
        elif latitude < self.s_temp:
            return ("s_temp")
        elif latitude < self.s_tundra: 
            return ("s_tundra")
        elif latitude < self.s_polar: 
            return ("s_polar")
        
    def set_terrain_type(self, i, j):
        temperature = self.temp_map[i][j]
        elevation = self.h_map[i][j]
        humidity = self.humid_map[i][j]
        climate_zone = self.get_climate_zone(i)
        slope = self.get_slope(i, j)

        if elevation < self.water_level:
            if temperature < self.freeze_pnt:
                return(ICE)
            else:
                return DEEP_OCEAN
        elif elevation < self.water_level: 
            if temperature < self.freeze_pnt:
                return(ICE)
            else:
                return OCEAN  
        
        if slope > 0.4:
            return(MOUNTAIN)
        
        if climate_zone == 'n_polar' or climate_zone == 's_polar':
            if elevation < self.water_level:
                if temperature < self.freeze_pnt:
                    return(ICE)
                else:
                    return(OCEAN)
            else:
                if temperature < self.freeze_pnt:
                    return(SNOW)
                else:
                    return(TUNDRA)    #CHANGE THIS TO GET SOMETHING ELSE THAN SNOW IN POLAR LAND
                
        if climate_zone == 'n_tundra' or climate_zone == 's_tundra':
            if temperature < self.freeze_pnt:
                return(SNOW)
            if humidity < 0.2:
                return(SAND)
            else:
                if elevation < self.beach_zone:
                    return(SAND)
                elif elevation < self.flatland_zone:
                    return(TUNDRA)
                elif elevation <= self.mountain_zone:
                    return(MOUNTAIN)
        
        if climate_zone == 'n_temp' or climate_zone == 's_temp':
            if temperature < self.freeze_pnt:
                return(SNOW)       
            if elevation < self.beach_zone:
                return(SAND)
            elif elevation < self.flatland_zone:
                if humidity < 0.2:
                    return(SAND)
                elif humidity < 0.4:
                    return(DRY_GRASS)
                elif humidity < 0.8:
                    return(GRASS)
                elif humidity < 1.0:
                    return(MARSHLAND)
            elif elevation <= self.mountain_zone:
                return(MOUNTAIN)
        if climate_zone == 'trop':
            if temperature < self.freeze_pnt:
                return(SNOW)       
            if elevation < self.beach_zone:
                return(SAND)
            elif elevation < self.flatland_zone:
                if humidity < 0.2:
                    return(SAND)
                elif humidity < 0.4:
                    return(SAVANNAH)
                elif humidity < 0.5:
                    return GRASS
                elif humidity < 1.0:
                    return(TROPICAL)
            elif elevation <= self.mountain_zone:
                return(MOUNTAIN)           
        
            

    #Transforms the height_map to a height_map shaped like a circle, with higher values closer to the center
    def circular_gradient_transform(self, h_map):
        center_j = self.height / 2
        center_i = self.length / 2
        grid = self.gen_meshgrid()
        disti = np.power(grid[1] - center_i,2)
        distj = np.power(grid[0] - center_j,2)
        distances = np.sqrt(disti + distj)

        distances = self.normalize_array(distances)
        h_map = h_map * distances
        h_map = self.normalize_array(h_map)
        return h_map
    
    def gen_meshgrid(self):
        i = np.arange(0,self.height)
        j = np.arange(0,self.length)
        grid = np.meshgrid(j,i)
        return grid
    
    #Define a number of states that can apply to a portion of the map
    #The map is divided into a grid
    #The state of each cell in the grid depends on neighbouring i-1 and j-1 states like a markov chain with probabilities of changing states
    def state_transform(self,grid_size, h_map):
        grid_i = grid_size[0]
        grid_j = grid_size[1]
        
        grid = np.zeros((self.height, self.length))
        
        for i in range(self.height):
            for j in range(self.length):
                if i > 0 and j > 0:
                    prev_i = grid[i-1][j]
                    prev_j = grid[i][j-1]
                    x = random.randint(0,1000)/1000.0
                    if prev_i == 0 and prev_j == 0:
                        if x < 0.995:
                            grid[i][j] = 0
                        else:
                            grid[i][j] = 1
                    if (prev_i == 0 and prev_j == 1) or (prev_i == 1 and prev_j == 0):
                        if x < 0.5:
                            grid[i][j] = 0
                        else:
                            grid[i][j] = 1    
                    else:
                        if x < 0.005:
                            grid[i][j] = 0
                        else:
                            grid[i][j] = 1
        h_map = h_map + grid
        h_map = self.normalize_array(h_map)
        return h_map
    #Temperature depends on elevation and latitude and month of the year
    #Temperature cycle over the year is modeled by a sine function
    def gen_temp_map(self, month):
        
        period_coeff = math.pi/6       #Period of the cycle is set to 12
        offset_coeff = 5                #Offset of sine function set to 4 to make the min value equal to 12
        temp_map = np.zeros((self.height, self.length))
        for i in range(self.height):
            for j in range(self.length):
                eq_dist = abs(i - self.height/2)
                #normalize eq distance
                eq_dist = eq_dist/(self.height * 0.5)
                temperature = (1 - self.h_map[i][j])/2 + (1 - eq_dist)/2
                temp_map[i][j] = temperature
        temp_map = self.normalize_array(temp_map)
        temp_map = temp_map * (0.2 * math.sin(offset_coeff + period_coeff * month) + 0.8)
        return temp_map

    #Converts a greyscale to a bluescale RGB color
    def grey_to_blue(self, grey_map):
        blue_map = np.zeros((self.length,self.height)+(3,))
        for i in range(self.height):
            for j in range(self.length):
                
                blue = 205 * grey_map[i][j] + 50
                rgb = [0,0,blue]
                blue_map[i][j] = rgb
        return blue_map
    
    #Convert greyscale to temperature scale colors
    def grey_to_temp(self, grey_map):
        temp_map = np.zeros((self.length,self.height)+(3,))
        for i in range(self.height):
            for j in range(self.length):        
                red = 205 * grey_map[i][j] + 50
                #blue = 255 * (1 - 255 * grey_map[i][j])
                rgb = [red,0,0]
                temp_map[i][j] = rgb
        return temp_map
    
    def gen_humid_map(self):
        humid_map = np.zeros((self.height, self.length))
        seed = random.randint(0,100)
        for i in range(self.height):
            for j in range(self.length):
                humid_map[i][j] = noise.pnoise2(i/self.scale, j/self.scale, octaves=6, persistence=0.6, lacunarity=self.lacunarity, repeatx=self.length, repeaty=self.height, base=seed)
        humid_map = self.normalize_array(humid_map)
        return humid_map
    
    #Get the slope of a point on the heightmap
    def get_slope(self, i ,j):
        
        if i >= self.height - 1 or j >= self.length - 1:    #Return zero for map edge
            return 0
        else:
            di = self.h_map[i + 1][j] - self.h_map[i][j]
            dj = self.h_map[i][j + 1] - self.h_map[i][j]
        
        return math.sqrt(di * di + dj * dj)
    
