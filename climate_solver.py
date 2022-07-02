
import noise
import random
import math
import numpy as np
import noisegen
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from numpy import inf
import climate_solver
from numba import njit
from PIL import Image
import cv2
from PIL import Image
from matplotlib import pyplot as plt

BLANK = [0,0,0]
DEEP_OCEAN = [30,75,235]
OCEAN = [65,105,225]
FORREST = [95,156,23]
TROPICAL = [29,120,88]
SAVANNAH = [204,217,65]
SAND = [238, 214, 175]
SNOW = [245, 245, 240]
ICE = [220,220,250]
MOUNTAIN = [139, 137, 137]
HIGH_MOUNTAIN = [99,95,95]
TUNDRA = [152,121,101]
BORREAL_FORREST = [33,82,16]
DRY_GRASS = [123,148,40]
MARSHLAND = [74,161,146]

COLORS = [DEEP_OCEAN,OCEAN, FORREST, TROPICAL, SAVANNAH,\
          SAND, SNOW, ICE, MOUNTAIN, TUNDRA, DRY_GRASS, MARSHLAND,\
          BORREAL_FORREST]


class World:

    """
    Class for the procedurally generated world

    """

    def __init__(self, height, length, periods):
        """
            Parameters
        ----------
        map_height : 
            Number of pixels of the map in the y direction
        map_length : 
            Number of pixels of the map in the y direction
        """
        self.height = height
        self.length = length
        self.periods = periods

        self.terrain = Terrain(self.height, self.length, self.periods)

class Terrain:

    """
    Terrain class for a procedurally generated world
    Climate class is instantiated for the terrain to simulate temperature, wind flow and humidity
    Together with the height map, the climate determines the biomes on the map
    """   
    def __init__(self, height, length, periods, octaves=6, persistence=0.61, \
                 lacunarity=2.3, water_lev=0.35, freeze_pnt=0.2):
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

        self.dt = 0.05
        self.seed = random.randint(0,10000)
        self.periods = periods

        self.height = height
        self.length = length

        #Parameters for height map
        self.scale = self.length / 4
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.gen_noise_maps

        #Terrain parameters
        self.water_lev = water_lev = water_lev
        self.deep_water_lev = self.water_lev - 0.15
        self.beach_zone = self.water_lev + 0.03
        self.freeze_pnt = freeze_pnt

        #Make terrain noise maps
        self.gen_noise_maps()

        self.climate = Climate(self.h_map, self.surf_h_map, (self.height, self.length), self.lat_map, self.lat_map_inv, self.dt, \
                                self.periods,self.land_map, water_level = self.water_lev)

        self.terrain_base = self.set_terrain_type()
        self.terrain_map = self.terrain_base
        self.terrain_set = self.generate_terrain()

    def generate_terrain(self):

        terrain_set = []

        for t in range(self.periods):
            nt = int(t / self.dt)
            temp_map = self.climate.temps[nt]
            self.terrain_map = self.apply_temp(temp_map, self.terrain_base)
            terrain_set.append(self.terrain_map)

        return terrain_set
            

    def gen_noise_maps(self):
        """
        Add a height map
        Add a surface height map which is the height map but with all values equal to water level below or equal to it
        Add a latitude map where the value equals 1 at the equator and 0 at the poles
        Add a inverted latitude map where the value equals 0 at the equator and 1 at the poles
        """
        #Generate height maps
        noise_map = noisegen.NoiseMap(self.height, self.length, exp_factor = 1.5)
        self.h_map = noise_map.gen_simplex_map(self.scale, self.octaves, self.persistence, self.lacunarity, self.seed)
        self.surf_h_map = self.surface_map(self.water_lev, self.h_map)
        
        #Generate latitude maps
        latmap = noisegen.LatMap(self.height, self.length)
        self.lat_map = latmap.gen_lat_map(invert = False)
        self.lat_map_inv = latmap.gen_lat_map(invert = True)

        #Generate slope maps
        self.slopes = np.gradient(self.h_map)
        self.slopes_x = self.slopes[0]
        self.slopes_y = self.slopes[1]

        #Generate land sea map
        l_map = noisegen.LandMap(self.height, self.length, self.h_map, self.water_lev)
        self.land_map = l_map.gen_land_map()


    def surface_map(self, level, amap):
        """
        The surface map is a map but with values under or equal to the level value equal to the level
        """
        surf_map = np.where(self.h_map <= level, level, amap)
        return surf_map

    #Turn a 2d array of floats into a 3d array to be compatible for masking
    def dimension_transform(self, amap):
        amap = np.reshape(amap, amap.shape +(1,))
        return amap

    #Apply temp map to terrain map to determine which areas are frozen (ice or snow)
    def apply_temp(self, temps, terrain_map):

        heights = self.h_map
        heights = self.dimension_transform(heights)
        temps = self.dimension_transform(temps)
        terrain_map = np.where((temps < self.freeze_pnt) & (heights > self.water_lev),SNOW,terrain_map)
        terrain_map = np.where((temps < self.freeze_pnt) & (heights <= self.water_lev),ICE,terrain_map)
        return terrain_map

    def set_terrain_type(self):
        """
        Assign terrain types (biomes) to the map based on the parameters:
            -temperature
            -elevation
            -humidity
        """
        
        temps = self.climate.avg_temp
        humidity = self.climate.avg_humidity
        terrain_map = np.zeros((self.height, self.length)+(1,))
        
        #Add extra placeholder dimension to temp, height, humidity and slope maps
        temps = self.dimension_transform(temps)
        heights = self.dimension_transform(self.h_map)
        humidity = self.dimension_transform(humidity)
        slope_x = self.dimension_transform(self.slopes_x)
        slope_y = self.dimension_transform(self.slopes_y)
        #rivers = self.dimension_transform(self.rivers)
        terrain_map = np.where(heights > 100,OCEAN,DRY_GRASS)
        #Determine tundra zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps < 0.2),TUNDRA,terrain_map)
        #Determine grassland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.2) & (humidity > 0.15),DRY_GRASS,terrain_map)
        #Determine temperate forrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.2) & (humidity > 0.2),FORREST,terrain_map)
        #Determine marshland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.2) & (humidity > 0.3),MARSHLAND,terrain_map)
        #Determine desert zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.25) & (humidity < 0.2),SAND,terrain_map)
        #Determine savannah zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.35) & (humidity >= 0.2),SAVANNAH,terrain_map)
        #Determine rainforrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.35) & (humidity > 0.3),TROPICAL,terrain_map)
        #Determine mountain zone by elevation
        terrain_map = np.where((heights > 0.6),MOUNTAIN,terrain_map)
        #Determine high mountain zone
        terrain_map = np.where((heights > 0.75),HIGH_MOUNTAIN,terrain_map)
        #Determine beach  zone
        terrain_map = np.where((heights >= self.water_lev) & (heights <= self.beach_zone) & (abs(slope_x) < 0.02) & (abs(slope_y) < 0.02),SAND,terrain_map)
        #Determine ocean  zone
        terrain_map = np.where((heights > self.deep_water_lev) & (heights < self.water_lev),OCEAN,terrain_map)
        #Determine deep ocean  zone
        terrain_map = np.where((heights < self.deep_water_lev),DEEP_OCEAN,terrain_map)
        #Plot rivers
        #terrain_map = np.where(rivers == 1, OCEAN, terrain_map)
        print(terrain_map.shape)
        return terrain_map



    #slope(gradient) map of a map
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

class Climate:

    def __init__(self, h_map,surf_h_map, res, lat_map, lat_map_inv, dt, periods,land_map, freeze_point = 0.13, water_level = 0.3):
        """
        The climate simulation can take a different set of heights and lengths to the original map to speed up computations
        The height map provided can be scaled down to fit resolution for the climate simulation
        Calculates wind flow (wind speeds and directions), temperature and humidity for the world
        """

        self.periods = periods
        self.dt = dt
        self.nx = res[1]
        self.ny = res[0]
        self.nt = int(periods / self.dt)
        self.res = res
        self.height = res[0]
        self.length = res[1]
        self.h_l_ratio = self.height / self.length
        self.water_level = water_level
        self.freeze_point = freeze_point
        self.h_map = h_map
        self.surf_h_map = surf_h_map
        self.lat_map = lat_map
        self.lat_map_inv = lat_map_inv
        self.land_map = land_map
        self.slopes = np.gradient(self.h_map)
        self.surf_slopes = np.gradient(self.surf_h_map)

        self.init_temp = self.gen_temp_map(0)
        

        self.windheat = self.wind_heat_sim()
        self.wind_x = self.windheat[0]
        self.wind_y = self.windheat[1]
        self.temps = self.windheat[2]

        #Calculate average humidity and temperatures
        self.avg_temp = self.avg_map(self.temps)
        self.init_evap = self.gen_evap_map()
        self.humid_evap = self.humid_sim()
        self.humidity = self.humid_evap[0]
        self.evap = self.humid_evap[1]
        self.avg_humidity = self.avg_map(self.humidity)#self.avg_map(self.humidity)
        
    def wind_heat_sim(self):
        """
        Create a setup for the much faster humidity simulation that utilizes Numba
        """
        g = 9.8     #Gravitational acceleration
        h = 5.0      #Temperature gradient parameter
        c = 0.15    #Coriolis effect paramter
        v = 5.0
        alpha = 0.2    #Temperature diffusivity parameter


        r1 = (self.dt * self.h_l_ratio)
        r2 = self.dt
        r3 = (self.dt * v * (self.h_l_ratio))
        r4 = (self.dt * v * (self.h_l_ratio))

        params = [g,h,c,r1,r2,r3,r4,alpha]

        nx = self.nx
        ny = self.ny
        nt = self.nt
        dt = self.dt

        wind = climate_solver.windheat_sim(nx, ny, nt, dt,self.slopes, self.surf_slopes, self.lat_map_inv,self.init_temp, self.h_map, params, self.land_map)

        #Normalize temperature maps
        for n in range(1,len(wind[2]) - 1):
            wind[2][n] = self.normalize_map(wind[2][n])

        return wind




    def humid_sim(self):
        """
        Create a setup for the much faster humidity simulation that utilizes Numba
        """
        k = 0.1 #Diffusion parameter
        p = 0.2
        d = 0.15   #Elevation parameter
        e = 0.25   #Evaporation parameter
        T = 0.6
        nx = self.nx
        ny = self.ny
        nt = self.nt
        dt = self.dt

        params = [k,p,d,e,self.h_l_ratio,T]
        slopes = np.gradient(self.surf_h_map)
        wind_x = self.wind_x
        wind_y = self.wind_y
        humidity = climate_solver.humidity_sim(wind_x, wind_y, nx, ny, nt, dt, self.init_evap, self.surf_h_map, self.temps, slopes,params) 
        print('test')
        #Normalize all humidity maps to 0-1
        for n in range(1,len(humidity) - 1):
            #humidity[n] = np.where(self.h_map < self.water_level, humidity[n], np.exp(humidity[n]))         #Amplify humidity over land
            humidity[n] = self.normalize_map(humidity[n])
            
        return humidity

    def gen_temp_map(self, t):
        """
        Temperature depends on elevation and latitude and month of the year
        Temperature cycle over the year is modeled by a sine function
        Total effect of latitude depends on a base latitude map that reduces temperature near the poles
        Two other latitude maps have highest values either at the south or the north pole
        Total effect of latitude is found by weighting north and south latitude effect based on month, then added to the base latitude
        TODO: Implement offset season for northern and southern hemisphere
        """
        #temp_map = np.where(self.h_map < self.water_level, self.water_level + 0.1, self.h_map)        #Transform ocean to uniform height

        temp_map = (self.surf_h_map - 1.0) * (-1.0)

        latmap = noisegen.LatMap(self.height, self.length)
        latitude_base = latmap.gen_lat_map()
        latitude_south = latmap.gen_lat_map(symmetric = False)
        latitude_south = latmap.invert_map(latitude_south)      #Invert one latitude map so that the poles will cycle
        latitude_north = latmap.gen_lat_map(symmetric = False)
        
        latitude_ns = np.sin(latitude_north * t/11 + latitude_south * (12 - t)/11)
        latitude = latitude_ns * 0.2 + latitude_base
        latitude = self.normalize_map(latitude)
        temp_map = (temp_map * latitude)
        
        #Even out temperature over the ocean
        #temp_map = np.where(self.h_map < self.water_level, temp_map - 0.05, temp_map)
        #Normalize
        temp_map = self.normalize_map(temp_map)
        return temp_map

    def gen_evap_map(self):
        """
        Evaporation depends on whether the point is below the water level (is ocean/water) and temperature
        """

        temp_map = self.avg_temp

        evap_map = np.zeros((self.height, self.length))
        evap_map = np.where(self.h_map <= self.water_level, 1.0 * temp_map, 0.95 * temp_map)        #Evaporation is strongest over the ocean
        evap_map = np.where(temp_map <= self.freeze_point, 0.15, evap_map)      #Evaporation is lower if below freezing
        evap_map = self.normalize_map(evap_map)
        return evap_map

    def normalize_map(self, amap):
        norm_map = (amap - amap.min()) / (amap.max() - amap.min())
        return norm_map

    def avg_map(self, map_set):
        """
        Calculate an average map from a set of maps, for example a sequence of temperature maps
        """
        avg_map = np.mean(map_set,axis = 0)
        print("Map shape:")
        print(avg_map.shape)
        return avg_map
