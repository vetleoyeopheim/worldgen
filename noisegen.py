# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:14:35 2021

@author: Vetle
"""


import random
import numpy as np
import noise
import math



points = 100


def gen_perlin_hmap(height, length, scale, octa, pers, lac, seed):
    h_map = np.zeros((height, length))
    for i in range(height):
        for j in range(length):
            h_map[i][j] = noise.pnoise2(i/scale, j/scale, octaves=octa, \
            persistence = pers, lacunarity=lac, repeatx=length, repeaty=height, base= seed)
    return h_map


