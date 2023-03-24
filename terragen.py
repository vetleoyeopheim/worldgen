
import numpy as np
import noisegen as ng
import random
import math
import climate_solver
import open3d
import colors
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib import pyplot as plt

COLORS = colors.COLORS


class Terrain:

    def __init__(self, size, water_level, periods = 365):
        """
        Parameters:
            -size: list of two integers. First component is the size of the y-axis(longitude) and the second component is the y-axis(latitude)

        """
        self.size_x = size[0]
        self.size_y = size[1]
        self.periods = periods
        self.water_level = water_level
        self.beach_zone = self.water_level + 0.03
        self.freeze_pnt = 0.25

        #Generate height map and slopes for the height map in the x and y direction
        self.create_height_map()
        self.slopes = self.create_slope_map(self.height_map)
        self.slope_x = self.slopes[0]
        self.slope_y = self.slopes[1]

        self.climate = Climate(size, self.height_map, 0.5,self.water_level,periods,self.slopes)

        self.base_temp = self.climate.init_temp
        self.base_humidity = self.climate.init_humidity

        self.biome_map = self.set_biomes_standard()

        self.terrain_maps = self.apply_temperatures()


    def create_height_map(self):
        scale = self.size_x / 4.0
        octaves = 6
        lacunarity = 2.3
        persistence = 0.61
        voronoi_factor = 0.45
        seed = random.randint(0,100000)
        noisemap = ng.NoiseMap(self.size_x, self.size_y)
        
        voronoi_map = noisemap.gen_voronoi_map(128)
        height_map = noisemap.gen_simplex_map(scale, octaves, persistence, lacunarity, seed)

        self.height_map = voronoi_factor * voronoi_map + height_map
        self.height_map = ((self.height_map - self.height_map.min()) / (self.height_map.max() - self.height_map.min()))
        
        self.sea_level_height_map = np.where(self.height_map <= self.water_level, self.water_level, self.height_map)
        #self.height_map = np.where(self.height_map <= self.water_level, self.water_level - 0.01, self.height_map)
        
    
    def apply_temperatures(self):
        """
        Generate a set of images where temperature has been applied for a full cycle (year)
        """

        terrain_maps = []
        
        heights = self.dimension_transform(self.height_map)

        for t in range(self.periods):
            #temps = self.dimension_transform(temps[t])
            temp = self.climate.temp_maps[t]
            temp = self.dimension_transform(temp)
            terrain_map = self.biome_map

            terrain_map = np.where((temp < self.freeze_pnt) & (heights > self.water_level),SNOW,terrain_map)
            terrain_map = np.where((temp < self.freeze_pnt) & (heights <= self.water_level),ICE,terrain_map)
            terrain_maps.append(terrain_map)

        return terrain_maps

    def create_slope_map(self, amap):
        slope_map = np.gradient(amap)
        
        return slope_map

    def set_biomes_standard(self):
        """
        Determine biome of the different parts of the world by using elevation, temperature and humidity
        """

        #temps = self.climate.temps[1]
        #humidity = self.climate.humidity[1]
        biome_map = np.zeros((self.size_x, self.size_y)+(1,))
        
        #Add extra placeholder dimension to temp, height, humidity and slope maps
        temps = self.dimension_transform(self.base_temp)
        heights = self.dimension_transform(self.height_map)
        humidity = self.dimension_transform(self.base_humidity)
        slope_x = self.dimension_transform(self.slope_x)
        slope_y = self.dimension_transform(self.slope_y)

        mountain_perim = 0.7        #Height where mountain zone starts
        hmountain_perim = 0.8      #Height where high mountain zone starts  
        
        tundra_temp_perim = 0.25    #Temperature below which terrain is tundra
        tropics_temp_perim = 0.4    #Temperature above which is tropical zone (savannah and tropical)

        desert_humid = 0.2      #Humidity value below which zone is desert
        forrest_humid = 0.3    #Humidity for forrest or savannah
        wetland_humid = 0.4   #Humidity above which we have wetlands (rainforest or marshland)
        tundra_humid_perim = 0.10

        deep_water_lev = 0.8 * self.water_level 

        biome_map = np.where(heights > 100,OCEAN,DRY_GRASS)

        biome_map = np.where((heights <= self.water_level),OCEAN, biome_map)
        #Determine beach  zone
        biome_map = np.where((heights >= self.water_level) & (heights <= self.beach_zone),SAND,biome_map)

        #Set arctic zones

        #Set cold zones, tundra for dry and cold and taiga for more humid and cold areas
        biome_map = np.where((heights > self.beach_zone) & (temps < tundra_temp_perim) & (humidity < tundra_humid_perim),TUNDRA,biome_map)
        biome_map = np.where((heights > self.beach_zone) & (temps < tundra_temp_perim) & (humidity >= tundra_humid_perim),BORREAL_FORREST,biome_map)

        #Set temperate zones: desert, forrest, grasslands and marshlands
        biome_map = np.where((heights > self.beach_zone) & (temps > tundra_temp_perim) & (humidity < desert_humid),SAND,biome_map)
        biome_map = np.where((heights > self.beach_zone) & (temps > tundra_temp_perim) & (humidity > desert_humid),DRY_GRASS,biome_map)
        biome_map = np.where((heights > self.beach_zone) & (temps > tundra_temp_perim) & (humidity > forrest_humid),FORREST,biome_map)
        biome_map = np.where((heights > self.beach_zone) & (temps > tundra_temp_perim) & (humidity > wetland_humid),MARSHLAND,biome_map)

        #Set tropics zones
        biome_map = np.where((heights > self.beach_zone) & (temps > tropics_temp_perim) & (humidity < desert_humid),SAND,biome_map)
        biome_map = np.where((heights > self.beach_zone) & (temps > tropics_temp_perim) & (humidity > forrest_humid),SAVANNAH,biome_map)
        biome_map = np.where((heights > self.beach_zone) & (temps > tropics_temp_perim) & (humidity > wetland_humid),TROPICAL,biome_map)

        #Determine mountain zone by elevation
        biome_map = np.where((heights > mountain_perim),MOUNTAIN,biome_map)
        #Determine high mountain zone
        biome_map = np.where((heights > hmountain_perim),HIGH_MOUNTAIN,biome_map)
        #Determine deep ocean  zone
        biome_map = np.where((heights < deep_water_lev),DEEP_OCEAN,biome_map)
        
        return biome_map


    #Turn a 2d array of floats into a 3d array to be compatible for masking
    def dimension_transform(self, amap):
        amap = np.reshape(amap, amap.shape +(1,))
        return amap



        

class Climate:

    def __init__(self, size, height_map, freeze_point, water_level, sim_length, slopes, deltas = [8,8]):
        
        self.size_x = size[0]
        self.size_y = size[1]
        self.deltas = deltas
        self.height_map = height_map
        self.slopes = slopes
        self.freeze_point = freeze_point
        self.water_level = water_level
        self.sim_length = sim_length            #Simulation length is in days

        self.create_latmap()
        
        self.init_temp = self.gen_init_temp()
        self.temp_maps = self.gen_temp_cycle()
        self.init_humidity = self.gen_init_humidity()

        self.downscale_simulation()

        self.wind_map = self.wind_sim()

    def downscale_simulation(self):
        """
        Scales down x and y grid of the simulation by dx and dy factors
        Used on initial temperatures and humidity
        """
        
        scaled_x = int(self.size_x / self.deltas[0])
        scaled_y = int(self.size_y / self.deltas[1])
        """
        img_temp = Image.fromarray(self.init_temp).convert('RGB')
        img_humid = Image.fromarray(self.init_humidity)
        img_heights = Image.fromarray(self.init_temp)

        img_temp = img_temp.resize((scaled_x, scaled_y))
        img_humid = img_temp.resize((scaled_x, scaled_y))
        img_heights = img_temp.resize((scaled_x, scaled_y))

        img_temp.save('scaled_temp.png')
        """
        self.temp_scaled = self.init_temp[::self.deltas[1], ::self.deltas[0]]
        self.humid_scaled = self.init_humidity[::self.deltas[1], ::self.deltas[0]]
        self.heights_scaled = self.height_map[::self.deltas[1], ::self.deltas[0]]
        self.lat_map_scaled = self.lat_map[::self.deltas[1], ::self.deltas[0]]
        self.lat_map_inv_scaled = self.lat_map_inv[::self.deltas[1], ::self.deltas[0]]

        print(self.temp_scaled.shape)

        plt.imsave('temp_scaled.png',self.temp_scaled, cmap = 'hot')

    def upscale_simulation(self):

        pass

        

    def wind_sim(self, scale_factor = 4, dt = 1.0, dx = 1.0, dy = 1.0):
        """
        Create a setup for wind simulation using Numba
        """
        nx = int(self.size_x / self.deltas[0])
        ny = int(self.size_y / self.deltas[1])
        lat_map_inv = self.lat_map_inv_scaled
        lat_map = self.lat_map_scaled
        height_map = self.heights_scaled
        temp = self.temp_scaled
        temp_gradient = np.gradient(temp)
        

        
        #Create a surface map with ocean level being flat
        cm = ClimateMap(nx, ny, height_map)
        surface_map = cm.surface_map(self.water_level)
        slopes = np.gradient(surface_map)
        land_map = cm.land_map(self.water_level)
    
        #wind = climate_solver.wind_heat_ns(nx, ny, dt, lat_map_inv, surface_map, temp, temp_gradient, slopes, lat_map, land_map)
        wind = climate_solver.wind_sim(nx, ny, dt, lat_map_inv, surface_map, temp_gradient, slopes)
        return wind

    def create_latmap(self):

        lmap = ng.LatMap(self.size_x, self.size_y)

        self.lat_map = lmap.gen_lat_map()        #Generate an inverted latitude map
        self.lat_map_inv = lmap.gen_lat_map(invert = False)



    def gen_init_temp(self):
        """
        Generate an initial temperature map based on latitude and a height map
        The initial temperature is around the autumnal equinox so that temperature is 
        """
        cm = ClimateMap(self.size_x, self.size_y, self.height_map)
        temp_map = cm.gen_temp_map(self.lat_map)
        #Normalize array
        temp_map = cm.normalize_map(temp_map)

        return temp_map

    def gen_init_humidity(self):
        """
        Generate an initial humidity map based on temperature and a height map
        """
        cm = ClimateMap(self.size_y, self.size_x, self.height_map)
        humid_Map = cm.gen_humid_map(self.init_temp)
        #Normalize array
        humid_Map = cm.normalize_map(humid_Map)

        return humid_Map

    def gen_temp_cycle(self):
        """
        Generate a temprature cycle for a year
        Returns an array of temp maps 
        """
        temp_maps = np.zeros((self.sim_length,self.size_x, self.size_y))
        latmap = ng.LatMap(self.size_x, self.size_y)
        temp_map = self.init_temp

        #Generate one sided latitude maps to be used for cycling the temperature
        season_map_s = latmap.gen_lat_map(symmetric = False, invert = True)
        season_map_n = latmap.gen_lat_map(symmetric = False, invert = False)
        plt.imsave('season_map_n.png', season_map_n)
        plt.imsave('season_map_s.png', season_map_s)
        #season_map_south = latmap.invert_map(season_map_north)

        #Define parameters used to create a sine function with period of 365 days and a peak at the autumnal equinox
        A = math.pi * 2 / 365
        B = 81.75
        B2 = 92.75
        #Paramaeter for the strength of the seasonal effect
        C = 0.25


        for t in range(self.sim_length):
            #season_map_t = 0.5 * (season_map_n * np.abs(90 - t) / 365 + season_map_s * np.abs(t - 270) / 365)
            #season_map_t = np.sin(A * (t - B)) * season_map_n + np.sin(A * (173 - t - B)) * season_map_s
            season_map_t = np.sin(A * (t - B)) * season_map_n + np.sin(A * (B - t)) * season_map_s
            #season_map_nt = np.sin(A * (t - B)) * season_map_n
            #season_map_t = season_map_nt + season_map_st
            #season_map_south_t = np.sin(A * (t - B)) * season_map_south
            
            season_map_t = self.normalize_array(season_map_t)

            temp_map = self.init_temp + C * season_map_t

            #temp_map = self.init_temp * season_map_t

            #Normalize temperature
            temp_map = self.normalize_array(temp_map)
            temp_maps[t] = temp_map

        self.season_map = season_map_t
        return temp_maps


    def normalize_array(self, array):
        array_norm = (array - array.min())/(array.max() - array.min())
        return array_norm

    def scale_array(self, factor, array):
        """
        Scales down the size of an array by a factor to create a more coarse grid for weather simulation
        If the original world size is 1024x512, and the factor is 4, then this will return an array that is 256x128
        Then creates an array of the original size 
        """
        x_y_ratio = self.size_x / self.size_y
        y_factor = int(factor / x_y_ratio)
        new_array = array[0:int(self.size_y / factor):y_factor, 0:int(self.size_x / factor):factor]
        
        return new_array

class ClimateMap():

    def __init__(self, size_x, size_y, height_map):

        self.height_map = height_map
        self.size_x = size_x
        self.size_y = size_y

    def normalize_map(self, amap):
        amap_norm = (amap - amap.min())/(amap.max() - amap.min())
        return amap_norm

    def gen_temp_map(self, lat_map):
        """
        Generate an initial temperature map based on latitude and a height map
        The initial temperature is around the autumnal equinox so that temperature is 
        """
        h_map = self.height_map
        h_map = (h_map - 1.0) * (-1.0)  #Invert height map for temperature calculation so that lowest points have the highest temperature
        lat_factor = 0.5       #Relative importance of latitude for temperature
        #temp_map = (1 - lat_factor) * h_map + lat_factor * lat_map
        temp_map = h_map * lat_map
        temp_map = gaussian_filter(temp_map, sigma = 1)     #Smooth temperature map with gaussian filter
        #Normalize array
        temp_map = self.normalize_map(temp_map)

        return temp_map

    def gen_humid_map(self, temp_map):
        """
        Generate an initial humidity map based on a height map and a latitude map, with equatorial latitutes having higher humidity
        """
        #humid_map = np.exp(temp_map)
        #humid_map = self.normalize_map(humid_map)
        rand_seed = random.randint(0,100000)
        noise_map = ng.NoiseMap(self.size_y, self.size_x)
        humid_map = noise_map.gen_simplex_map(scale = self.size_x / 4.0, octa = 5, pers = 0.6, lac = 2.3, seed = rand_seed)
        lat_map = ng.LatMap(self.size_y, self.size_x)
        latitudes = lat_map.gen_lat_map()
        humid_map = humid_map * latitudes

        return humid_map

    def surface_map(self, level):
        """
        The surface map is a map but with values under or equal to the level value equal to the level
        """
        surf_map = np.where(self.height_map <= level, level, self.height_map)
        return surf_map

    def land_map(self, level):
        """
        Create a map that is equal to 1 over a given level and 0 under
        Used to create land ocean binary maps for example
        """
        land_map = self.height_map
        land_map = np.where(land_map > level, 1, 0)
        return land_map

    
