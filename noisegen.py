# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:14:35 2021

@author: Vetle
"""

from matplotlib import pyplot as plt
import random
import numpy as np
import noise
import math

class NoiseMap:
    """
    Parameters:
        algo: currently only perlin noise
        wrapping: either cylinder or plane. Cylinder makes the noise wrap around itself, but is TODO for now
        
    """
    def __init__(self, height, length, algo = 'perlin', wrapping = 'plane'):
        self.height = height
        self.length = length
        self.wrapping = wrapping


    def gen_perlin_map(self,scale,octa, pers, lac, seed):
        h_map = np.zeros((self.height, self.length))
        for i in range(self.height):
            for j in range(self.length):
                coords = (i,j)
                z = noise.snoise2(coords[0]/scale, coords[1]/scale, octaves=octa, \
                persistence = pers, lacunarity=lac, repeatx=self.height, repeaty = self.length, base= seed)
                h_map[i][j] = z
        h_map = self.exp_transform(h_map)
        #h_map = self.arctan_transform(h_map)
        h_map = self.normalize_array(h_map)
        return h_map
    
    def cyl_transform(self, i,j):
        """
        Applies a cylindrical transform to the noise map so that it will wrap around itself
        """
        tau = 2 * math.pi
        angle_x = tau * i
        return (math.cos(angle_x)/tau,math.sin(angle_x)/tau,j) 


        #Transforms the height_map to a height_map shaped like a circle, with higher values closer to the center
        #TODO: This needs to be fixed
    def circular_gradient_transform(self, h_map, height, length):
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

    #Exponential transform for noise function
    def exp_transform(self,nmap):
        nmap = np.exp(1.2 * nmap)
        return nmap
    
    def arctan_transform(self, nmap):
        nmap = np.arctan(nmap/math.pi)
        return nmap
    
    def normalize_array(self, array):
        array_norm = (array - array.min())/(array.max() - array.min())
        return array_norm

class LatMap():
    """
    Latitude map takes height and length to calculate an array with a distance to the equator
    """
    def __init__(self, height, length):

        self.height = height
        self.length = length

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
        
        lat_map = (lat_map - 1) * (-1)          #This inverts the latitude map so that the equator equals 1 and the poles equals zero
        lat_map = (lat_map - lat_map.min())/(lat_map.max() - lat_map.min())     
        return lat_map
