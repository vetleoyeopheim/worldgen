# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:16:02 2021

@author: Vetle
"""

import numpy as np
import terragen
import time
from PIL import Image
from matplotlib import pyplot as plt
start = time.time()
print("start")


world = terragen.World(512,1024)
terrain = world.gen_terrain_map()
hmap = world.temp_map
plt.imshow(hmap)
img = Image.fromarray(terrain.astype(np.uint8))
img.show()
end = time.time()
print(end - start)