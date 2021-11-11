
"""
@author: Vetle
"""


import numpy as np
import terragen
import time
from PIL import Image
import noisegen


from matplotlib import pyplot as plt
start = time.time()
print(r"C:\Users\Vetle\Documents\Python\Visual Studio\worldgen\worldgen\worldgen")

T = 1
t = 1
world = terragen.Terrain(512,1024)

path = ""


while t <= T:
    
    h_map = world.h_map
    terrain = world.apply_temp(t)
    temp_map = world.gen_temp_map(t)
    #h_map = world.h_map
    #temp_map = world.grey_to_temp(temp_map)

    img_terr = Image.fromarray(terrain.astype(np.uint8))
    img_temp = Image.fromarray(temp_map * 255)
    img_height = Image.fromarray(h_map * 255)
    img_temp = img_temp.convert('RGB')

    #img_temp = Image.fromarray(temp_map * 255)
    
    img_terr.save(path + "terr" + str(t) + ".png")
    img_temp.save(path + "temp" + str(t) + ".png")
    #img_height.save(path + "height" + str(t) + ".png")
    t += 1
    print(t)




end = time.time()
print(end - start)
