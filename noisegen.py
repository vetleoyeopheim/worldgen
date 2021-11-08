# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:14:35 2021

@author: Vetle
"""


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

    def gen_perlin_hmap(self, scale, octa, pers, lac, seed):
        h_map = np.zeros((self.height, self.length))
        for i in range(self.height):
            for j in range(self.length):
                if self.wrapping == 'cylinder':
                    coords = self.cyl_transform(i,j)
                    z = noise.pnoise3(coords[0]/scale, coords[1]/scale,coords[2]/scale, octaves=octa, \
                    persistence = pers, lacunarity=lac, repeatx=self.length, repeaty=self.height, repeatz = 1, base= seed)
                else:
                    coords = (i,j)

                    z = noise.pnoise2(coords[0]/scale, coords[1]/scale, octaves=octa, \
                    persistence = pers, lacunarity=lac, repeatx=self.height, base= seed)
                h_map[i][j] = z
        h_map = self.exp_filter(h_map)
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

    #Exponential filter for noise function
    def exp_filter(self,h_map):
        h_map = np.exp(1.5 * h_map)
        return h_map
                
    
    def normalize_array(self, array):
        array_norm = (array - array.min())/(array.max() - array.min())
        return array_norm


class LatMap():
    """
    Possible future latitude map class
    """
    def __init__(self, heigth, length):

        self.height = height
        self.length = length


    
