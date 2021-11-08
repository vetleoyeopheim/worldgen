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
    
    def __init__(self, height, length, algo = 'perlin'):
        self.height = height
        self.length = length

    def gen_perlin_hmap(self, scale, octa, pers, lac, seed):
        h_map = np.zeros((self.height, self.length))
        for i in range(self.height):
            for j in range(self.length):
                h_map[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octa, \
                persistence = pers, lacunarity=lac, repeatx=self.length, repeaty=self.height, base= seed)
        h_map = self.exp_filter(h_map)
        h_map = self.normalize_array(h_map)
        return h_map
    
    
        #Transforms the height_map to a height_map shaped like a circle, with higher values closer to the center
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
        h_map = np.exp(h_map)
        return h_map
                
    
    def normalize_array(self, array):
        array_norm = (array - array.min())/(array.max() - array.min())
        return array_norm
    
