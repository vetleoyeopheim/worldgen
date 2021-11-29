
"""
@author: Vetle Øye Opheim
"""

import noise
import random
import math
import numpy as np
import noisegen
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from numpy import inf
import climate_solver


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


class Terrain:
    
    """
    Terrain class for a procedurally generated world
    Climate class is instantiated for the terrain to simulate temperature, wind flow and humidity
    Together with the height map, the climate determines the biomes on the map
    """   
    def __init__(self, map_height, map_length, octaves=6, persistence=0.61, \
                 lacunarity=2.3, water_level=0.4, freeze_pnt=0.16):
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
        mode: optional
            Determine whether the climate for the terrain shall be simulated sequantially or i a coupled manner
            In the sequential mode, temperature is calculated independently of wind flow and humidity
            In coupled mode temperature is recalculated as the wind flow of the world evolves
            
        """
        self.dt = 0.005
        self.periods = 10
        self.height = map_height
        self.length = map_length
        self.scale = self.length / 4
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        
        self.water_level = water_level
        self.d_water_level = self.water_level - 0.15         #Deep water
        self.freeze_pnt = freeze_pnt
        
        self.beach_zone = self.water_level + 0.03
        self.seed = random.randint(0,1000)

        self.h_map = self.gen_hmap(self.octaves)
        self.lat_map = self.gen_lat_map()
        self.lat_map_inv = self.gen_lat_map(invert = False)
        self.surf_h_map = self.surface_hmap()
        
        self.add_climate()
        
        self.humid_map = self.humidity[29]


        self.slope_map = self.get_slope_map(self.h_map)
        self.terr_base = self.gen_terrain_map()         #Basis terrain before ice and snow (temperature effects) are added

    def add_climate(self):
        self.climate = Climate(self.h_map, self.surf_h_map, (self.height, self.length), self.lat_map, self.lat_map_inv, self.dt, self.periods, water_level = self.water_level, mode = 'sequential')
        self.wind = self.climate.wind
        self.temps = self.climate.temps
        self.temp_map = self.temps[3]
        self.humidity = self.climate.humidity
    
    def land_humidity(self, amap):
        """
        Transform humid map into land based map that sets ocean humidity equal to zero
        """
        humid_map = amap
        humid_map = np.where(self.h_map <= self.water_level, 0, humid_map)
        humid_map = normalize_map(humid_map)
        return humid_map

    def gen_lat_map(self, invert = True):
        """
        Setting invert = False generates a latitude map where the equator equals one and the poles equals zero
        """
        lat_map = noisegen.LatMap(self.height, self.length)
        latitude = lat_map.gen_lat_map(invert = invert)

        return latitude
    
    def create_clouds(self, month):
        """
        Clouds are modeled to be formed where humidity is high relative to temperature
        """
        sat_point = 0.5
        clouds = self.humidity[month] / (self.climate.temps[month] + 0.01)
        clouds = normalize_map(clouds)
        clouds = np.where(clouds > sat_point, clouds * 0.5, 0)
        
        return clouds

    def gen_hmap(self, octaves):
        noise_map = noisegen.NoiseMap(self.height, self.length, exp_factor = 1)
        h_map = noise_map.gen_simplex_map(self.scale, octaves, self.persistence, self.lacunarity, self.seed)
        
        return h_map

    def surface_temp_map(self, temp_map):
        """
        Create a temperature map which ignores ocean surface temperature
        """
        surf_temp_map = np.where(self.h_map <= self.water_level, 0, temp_map)
        surf_temp_map = normalize_map(surf_temp_map)
        return surf_temp_map

    def surface_hmap(self):
        """
        The surface height map is the regular height map but with heights over water equal to the water level
        This ensures that the oceans are completely flat, and can thereby be used for calculations of wind flow
        """
        surf_hmap = np.where(self.h_map <= self.water_level, self.water_level, self.h_map)
        return surf_hmap
                
    #Assign colors to map based on height_map and longitude()
    def gen_terrain_map(self):
        t_map = self.set_terrain_type()
        return t_map
    
    #Turn a 2d array of floats into a 3d array to be compatible for masking
    def dimension_transform(self, amap):
        amap = np.reshape(amap, amap.shape +(1,))
        return amap

    #Apply temp map to terrain map to determine which areas are frozen (ice or snow)
    def apply_temp(self, t):
        terrain_map = self.terr_base
        t = t % 11
        temps = self.climate.temps[t]
        temps = self.dimension_transform(temps)
        heights = self.h_map
        heights = self.dimension_transform(heights)
        terrain_map = np.where((temps < self.freeze_pnt) & (heights > self.water_level),SNOW,terrain_map)
        terrain_map = np.where((temps < self.freeze_pnt) & (heights <= self.water_level),ICE,terrain_map)
        return terrain_map

    def set_terrain_type(self):
        """
        Assign terrain types (biomes) to the map based on the parameters:
            -temperature
            -elevation
            -humidity
        """
        
        temps = self.temp_map    #Temperature in september as the "average" temperature of the year is what will determine the biome
        heights = self.h_map
        humidity = self.humid_map
        terrain_map = np.zeros((self.height, self.length)+(1,))

        #Add extra placeholder dimension to temperature map
        temps = self.dimension_transform(temps)
        heights = self.dimension_transform(heights)
        humidity = self.dimension_transform(humidity)

        terrain_map = np.where(heights > 100,OCEAN,DRY_GRASS)
        #Determine tundra zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps < 0.2),TUNDRA,terrain_map)
        #Determine grassland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.2) & (humidity > 0.15),DRY_GRASS,terrain_map)
        #Determine temperate forrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.2) & (humidity > 0.20),FORREST,terrain_map)
        #Determine marshland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.2) & (humidity > 0.30),MARSHLAND,terrain_map)
        #Determine desert zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity < 0.22),SAND,terrain_map)
        #Determine savannah zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity >= 0.22),SAVANNAH,terrain_map)
        #Determine rainforrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity > 0.35),TROPICAL,terrain_map)
        #Determine mountain zone
        terrain_map = np.where((heights > 0.6),MOUNTAIN,terrain_map)
        #Determine high mountain zone
        terrain_map = np.where((heights > 0.75),HIGH_MOUNTAIN,terrain_map)
        #Determine beach  zone
        terrain_map = np.where((heights >= self.water_level) & (heights <= self.beach_zone),SAND,terrain_map)
        #Determine ocean  zone
        terrain_map = np.where((heights > self.d_water_level) & (heights < self.water_level),OCEAN,terrain_map)
        #Determine deep ocean  zone
        terrain_map = np.where((heights < self.d_water_level),DEEP_OCEAN,terrain_map)
        
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

    def __init__(self, h_map,surf_h_map, res, lat_map, lat_map_inv, dt, periods, freeze_point = 0.13, water_level = 0.3, mode = 'coupled'):
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
        self.mode = mode
        self.slopes = np.gradient(self.h_map)

        if self.mode == 'sequential':
            self.temps = self.annual_temp_cycle()
            self.temp_grads = self.calc_temp_grads()
            self.evaps = self.annual_evap_cycle()
            self.evap_map = self.evaps[0]
            self.wind = self.fast_wind_sim()
            self.wind_x = self.wind[0]
            self.wind_y = self.wind[1]
            self.humidity = self.fast_humid_sim()

        elif self.mode == 'coupled':
            self.init_temp = self.gen_temp_map(1)

            self.wind_temp = self.fast_wt_sim()     #Simulating temperature as a coupled variable with wind flow is still experimental
            self.wind_x = self.wind_temp[0]
            self.wind_y = self.wind_temp[1]
            self.wind = np.array((self.wind_x,self.wind_y))
            self.temps = self.wind_temp[2]
            self.evaps = self.annual_evap_cycle()
            self.evap_map = self.evaps[0]
            print('test')
            self.humidity = self.fast_humid_sim()
        
    
    def fast_wt_sim(self):
        """
        Create a setup for the much faster humidity simulation that utilizes Numba
        """
        g = 9.8     #Gravitational acceleration
        h = 8.0       #Temperature gradient parameter
        c = 0.25    #Coriolis effect paramter
        v = 5.0   #Wind viscosity parameter
        alpha = 0.1       #Temperature diffusivity parameter


        r1 = (self.dt * self.h_l_ratio)
        r2 = self.dt
        r3 = (self.dt * v * (self.h_l_ratio))
        r4 = (self.dt * v * (self.h_l_ratio))

        params = [g,h,c,r1,r2,r3,r4,alpha]

        nx = self.nx
        ny = self.ny
        nt = self.nt
        dt = self.dt

        wind = climate_solver.windheat_sim(nx, ny, nt, dt, self.slopes, self.lat_map_inv,self.init_temp, self.h_map, params)

        return wind


    def annual_temp_cycle(self):
        """
        Generate temperature maps to represent a annual temperature cycle
        """
        temps = np.zeros((12, self.ny, self.nx))

        for t in range(0, 12, 1):
            temp_map = self.gen_temp_map(t)
            temps[t] = temp_map
            print('t')
        return temps

    def annual_evap_cycle(self):
        """
        Generate evap maps to represent an annual evap cycle
        """
        evap = np.zeros((12, self.ny, self.nx))

        for t in range(0, 12, 1):
            evap_map = self.gen_evap_map(t)
            evap[t] = evap_map
        print(evap.shape)
        return evap

    def calc_temp_grads(self):
        temp_grads = np.zeros((self.nt,2, self.ny, self.nx))
        for t in range(0,12,1):
            temp = self.temps[t]
            temp_grads[t] = np.gradient(temp)
        
        return temp_grads

    def fast_temp_sim(self, init_temps):
        """
        The temp sim lets the heat map diffuse a bit 
        This is still experimental
        """
        dt = 0.1       #Define the dt seperately for this equation
        alpha = 0.001     #Diffusion coefficient
        r1 = (dt * self.h_l_ratio)
        r2 = dt
        periods = 2

        params = [alpha, r1, r2]

        temperatures = climate_solver.temp_sim(self.nx, self.ny, periods, dt, init_temps, params)
        return temperatures

    def fast_wind_sim(self):
        """
        Create a setup for the much faster humidity simulation that utilizes Numba
        """
        g = 9.8     #Gravitational acceleration
        h = 8.0       #Temperature gradient parameter
        c = 0.25    #Coriolis effect paramter
        v = 5.0

        r1 = (self.dt * self.h_l_ratio)
        r2 = self.dt
        r3 = (self.dt * v * (self.h_l_ratio))
        r4 = (self.dt * v * (self.h_l_ratio))

        params = [g,h,c,r1,r2,r3,r4,v]

        nx = self.nx
        ny = self.ny
        nt = self.nt
        dt = self.dt
        temps = self.temps
        latitudes = self.lat_map_inv

        wind = climate_solver.wind_sim(nx, ny, nt, dt, self.slopes, self.temp_grads, latitudes, params)
        return wind

    def fast_humid_sim(self):
        """
        Create a setup for the much faster humidity simulation that utilizes Numba
        """
        k = 0.1 #Diffusion parameter
        p = 1.0
        d = 0.2   #Elevation parameter
        e = 0.5   #Evaporation parameter
        T = 0.8
        nx = self.nx
        ny = self.ny
        nt = self.nt
        dt = self.dt

        params = [k,p,d,e,self.h_l_ratio,T]
        slopes = np.gradient(self.surf_h_map)
        wind_x = self.wind[0]
        wind_y = self.wind[1]
        humidity = climate_solver.humidity_sim(wind_x, wind_y, nx, ny, nt, dt, self.evaps, self.surf_h_map, self.temps, slopes,params) 
        print('test')
        #Normalize all humidity maps to 0-1
        for n in range(1,len(humidity) - 1):
            #humidity[n] = np.where(self.h_map < self.water_level, humidity[n], np.exp(humidity[n]))         #Amplify humidity over land
            humidity[n] = normalize_map(humidity[n])
            
        return humidity

    def gen_evap_map(self, t):
        """
        Evaporation depends on whether the point is below the water level (is ocean/water) and temperature
        """
        if self.mode == 'sequential':
            temp_map = self.temps[t]

        elif self.mode == 'coupled':
            temp_map = self.init_temp
        evap_map = np.zeros((self.height, self.length))
        evap_map = np.where(self.h_map <= self.water_level, 1.0, 0.2)        #Evaporation is strongest over the ocean
        evap_map = np.where(temp_map <= self.freeze_point, 0.01, evap_map)      #Evaporation is lower if below freezing

        evap_map = evap_map * np.exp(temp_map)
        #evap_map = np.exp(evap_map)
        evap_map = normalize_map(evap_map)
        return evap_map
    
   
    def recalc_evap_map(self):
        self.evap_map = self.gen_evap_map()

    
    def avg_humidity(self):
        """
        Calculate a single humidity map as the average of all simulated humidity maps in the humidity array
        """
        avg_humidity = self.humidity[0]
        N = len(self.humidity)
        for n in range(1,N, 1):
            avg_humidity = avg_humidity + self.humidity[n]

        avg_humidity = avg_humidity / N
        return avg_humidity

    def rainfall_sim(self):
        """
        Rainfall is calculated as the change in humidity over the map
        NOT YET IMPLEMENTED
        """
        pass
             
    def recalc_temperature(self, month):
        """Also recalculates the temperature gradient"""
        self.temp_map = self.gen_temp_map(month)
        self.temp_grad = np.gradient(self.temp_map)
    
    
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

        temp_map = (self.h_map - 1.0) * (-1.0)
        latmap = noisegen.LatMap(self.height, self.length)
        latitude_base = latmap.gen_lat_map()
        latitude_south = latmap.gen_lat_map(symmetric = False)
        latitude_south = latmap.invert_map(latitude_south)      #Invert one latitude map so that the poles will cycle
        latitude_north = latmap.gen_lat_map(symmetric = False)
        
        #latitude_ns = latitude_north * (t/12) + latitude_south * ((11 - t) / 12)
        latitude_ns = np.sin(latitude_north * t/11 + latitude_south * (12 - t)/11)
        latitude = latitude_ns * 0.3 + latitude_base
        latitude = normalize_map(latitude)
        temp_map = (temp_map * latitude)
        
        #Even out temperature over the ocean
        #temp_map = np.where(self.h_map < self.water_level, temp_map - 0.05, temp_map)
        #Normalize

        #Run the temperatues through a diffusion process (but not working atm.)
        #temp_map = self.fast_temp_sim(temp_map)
        temp_map = normalize_map(temp_map)
        return temp_map

    def gen_humid_map_simple(self):
        """
        Generates a humidity map based on the height map
        Humidity is inversely related to height so that areas with ocean or close to the ocean gets more humidity
        """
        humid_map = np.exp((self.h_map - 1) * (-1))

        #Normalize
        humid_map = (humid_map - humid_map.min())/(humid_map.max() - humid_map.min())

        return humid_map
    
def normalize_map(amap):
    norm_map = (amap - amap.min()) / (amap.max() - amap.min())
    
    return norm_map

