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
    def __init__(self, map_height, map_length, octaves=9, persistence=0.6, \
                 lacunarity=2.1, water_level=0.30, freeze_pnt=0.14):
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
        self.scale = self.length / 2
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
            
        self.water_level = water_level
        self.d_water_level = self.water_level - 0.1         #Deep water
        self.freeze_pnt = freeze_pnt
        
        self.beach_zone = self.water_level + 0.03
        self.seed = random.randint(0,1000)

        self.h_map = self.gen_hmap()
        self.lat_map = self.gen_lat_map()
        self.surf_hmap = self.surface_hmap()
        
        self.climate = Climate(self.h_map, (self.height, self.length), self.lat_map, water_level = self.water_level)
        self.temp_map = self.climate.gen_temp_map(3)        #Temp map in month 3 (april) is taken as the average and used for calculations of biomes
        self.humidity = self.climate.humidity
        self.wind = self.climate.wind
        self.humid_map = self.climate.avg_humidity()
        self.humid_map = (self.humid_map - self.humid_map.min() / (self.humid_map.max() - self.humid_map.min()))
        self.slope_map = self.get_slope_map(self.h_map)
        self.terr_base = self.gen_terrain_map()         #Basis terrain before ice and snow (temperature effects) are added
        #self.terrain_map = self.apply_temp(12)
    
    def gen_lat_map(self):
        lat_map = noisegen.LatMap(self.height, self.length)
        latitude = lat_map.gen_lat_map()
        return latitude
    
    def gen_hmap(self):
        noise_map = noisegen.NoiseMap(self.height, self.length)
        h_map = noise_map.gen_simplex_map(self.scale, 7, 0.6, 2.1, self.seed)
        
        return h_map

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

    #Apply temp map to terrain map to determine which
    def apply_temp(self, month):
        terrain_map = self.terr_base
        temps = self.climate.temp_map
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
        
        temps = self.climate.gen_temp_map(3)        #Temperature in september as the "average" temperature of the year is what will determine the biome
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
        #Determine borreal forrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3),BORREAL_FORREST,terrain_map)
        #Determine grassland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity < 0.25),DRY_GRASS,terrain_map)
        #Determine temperate forrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity > 0.25),FORREST,terrain_map)
        #Determine marshland zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity > 0.4),MARSHLAND,terrain_map)
        #Determine desert zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.3) & (humidity < 0.2),SAND,terrain_map)
        #Determine savannah zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.50) & (humidity > 0.2),SAVANNAH,terrain_map)
        #Determine rainforrest zone
        terrain_map = np.where((heights > self.beach_zone) & (heights < 0.6) & (temps > 0.50) & (humidity > 0.4),TROPICAL,terrain_map)
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
    
    #slope(gradient) map of a height map
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

    def __init__(self, h_map, res, lat_map, freeze_point = 0.13, water_level = 0.3):
        """
        The climate simulation can take a different set of heights and lengths to the original map to speed up computations
        The height map provided can be scaled down to fit resolution for the climate simulation
        Calculates wind flow (wind speeds and directions), temperature and humidity for the world
        """
        self.res = res
        self.height = res[0]
        self.length = res[1]
        self.h_l_ratio = self.height / self.length
        self.water_level = water_level
        self.freeze_point = freeze_point
        self.h_map = h_map
        self.lat_map = lat_map
        self.temp_map = self.gen_temp_map(9)
        self.temp_grad = np.gradient(self.temp_map)
        self.slopes = np.gradient(h_map)
        
        self.evap_map = self.gen_evap_map()
        self.dt = 0.05
        self.nx = res[1]
        self.ny = res[0]
        self.nt = int(5.0 / self.dt)

        #self.wind = self.wind_sim(5, 0.05, 10)
        self.wind = self.fast_wind_sim()
        #self.humidity = self.humidity_sim(self.wind[0], self.wind[1], 5, 0.05)
        self.humidity = self.fast_humid_sim()
    
    def fast_wind_sim(self):
        """
        Create a setup for the much faster humidity simulation that utilizes Numba
        """
        g = 9.8     #Gravitational acceleration
        h = 10.0       #Temperature gradient parameter
        c = 0.27    #Coriolis effect paramter
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
        slopes = self.slopes
        temp_grad = self.temp_grad
        latitudes = latitude = (self.lat_map - 1) * (-1)

        wind = climate_solver.wind_sim(nx, ny, nt, dt, slopes, temp_grad, latitudes, params)
        return wind

    def fast_humid_sim(self):
        """
        Create a setup for the much faster humidity simulation that utilizes Numba
        """
        k = 0.1   #Diffusion parameter
        p = 1.3
        d = 0.4     #Elevation parameter
        e = 0.4     #Evaporation parameter

        nx = self.nx
        ny = self.ny
        nt = self.nt
        dt = self.dt

        params = [k,p,d,e,self.h_l_ratio]
        slopes = self.slopes
        evap_map = self.evap_map
        h_map = self.h_map
        wind_x = self.wind[0]
        wind_y = self.wind[1]
        humidity = climate_solver.humidity_sim(wind_x, wind_y, nx, ny, nt, dt, evap_map, h_map, params) 
        #Normalize all humidity maps to 0-1 range
        for n in range(1,len(humidity) - 1):
            humidity[n] = normalize_map(humidity[n])
        return humidity

    def gen_evap_map(self):
        """
        Evaporation depends on whether the point is below the water level (is ocean/water) and temperature
        """
        evap_map = np.zeros((self.height, self.length))
        evap_map = np.where(self.h_map <= self.water_level, 1, 0.25)        #Evaporation is strongest over the ocean
        evap_map = np.where(self.temp_map <= self.freeze_point, 0.05, evap_map)      #Evaporation is lower if below freezing
        evap_map = evap_map * self.temp_map
        evap_map = normalize_map(evap_map)
        return evap_map
    
   
    def recalc_evap_map(self):
        self.evap_map = self.gen_evap_map()


    def humidity_sim(self, wind_x, wind_y, t_range, dt):
        """
        NOTE: THIS RUNS THE HUMIDITY SIMULATION WITHOUT NUMBA, AND IS SIGNIFICANTLY SLOWER. USE THE "climate_solver" version instead
        Humidity depends on wind direction, wind strength and evaporation
        Strong winds carry humidity away while evaporation increases humidity on a given point
        Humidity also drops as elevation increases
        """
        k = 1.0     #Diffusion parameter
        p = 0.9
        d = 0.3     #Elevation parameter
        e = 0.7     #Evaporation parameter
        nt = int(t_range / dt)
        humidity = np.zeros((nt,self.ny,self.nx))
        humidity[0] = self.evap_map     #Initial condiation is just the evaporation
        self.recalc_temperature(0)
        for t in range(1,nt,1):
            #Recalculate temperature if needed
            t_m = t * dt
            if float(t_m).is_integer():
                self.recalc_temperature(t_m)
                self.recalc_evap_map()
            xy_arr = np.zeros((self.ny, self.nx))
            for i in range(0,self.ny,1):
                x_col = np.zeros(self.nx)
                for j in range(0,self.nx,1):
                    """
                    h = humidity[t - 1][i][j] + dt * e * evap[i][j] - \
                        dt * k * math.sqrt(wind_x[t - 1][i][j] **2 + wind_y[t - 1][i][j] ** 2)\
                            - dt * d * self.h_map[i][j]
                    """
                    if i == self.ny - 1:
                        h = 0
                    elif j == self.nx - 1:
                        h = 0
                    else:
                        h = humidity[t - 1][i][j]\
                            - dt * p * self.h_l_ratio * wind_x[t - 1][i][j] * (humidity[t - 1][i][j] - humidity[t - 1][i - 1][j])\
                            - dt * p * wind_y[t - 1][i][j] * (humidity[t - 1][i][j] - humidity[t - 1][i][j - 1])\
                            + dt * k * (self.h_l_ratio **2) * (humidity[t - 1][i + 1][j] + humidity[t - 1][i - 1][j] - 2 * humidity[t - 1][i][j])\
                            + dt * k * (humidity[t - 1][i][j + 1] + humidity[t - 1][i][j - 1] - 2 * humidity[t - 1][i][j])\
                            + dt * e * self.evap_map[i][j]\
                            - dt * d * self.h_map[i][j]
                   
                    x_col[j] = h
                xy_arr[i] = x_col
            humidity[t] = xy_arr
            print(t)
        #Normalize all humidity maps to 0-1 range
        for n in range(1,len(humidity) - 1):
            humidity[n] = normalize_map(humidity[n])
        return humidity
    
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
    
    def wind_sim(self, t_range, dt, v):
        """
        NOTE: THIS RUNS THE WIND SIMULATION WITHOUT NUMBA, AND IS SIGNIFICANTLY SLOWER. USE THE "climate_solver" version instead
        Calculate a wind flow map based on a 2D Burgers equation
        Calculations are based on the height map. A negative slope accelerates wind speed due to gravity (g). Opposite is true for upward facing slopes
        Wind speed and direction also depends on temperature gradient.
        A positive temperature gradient (cold air goes to warm air) increases wind speeds
        A latitude term is added to the x-direction of the simulation to mimic a coriolis type effect
        Equations are:
            dv_dx = -du_dy - v * (d2v/dx2 + d2v/dy) - g * s_x(x,y) + h * tg_x(x,y) + latitude(x,y)
            du_dy = -dv_dx - v * (d2u/dx2 + d2u/dy) - g * s_y(x,y) + h * tg_y(x,y)
        Where g is gravitational acceleration, and s_x and s_y are the gradients in the x and y directions. 
        tg is the temperature gradient
        """

        nt = int(t_range / dt)
        
        r1 = (dt * self.h_l_ratio)
        r2 = dt
        r3 = (dt * v * (self.h_l_ratio))
        r4 = (dt * v * (self.h_l_ratio))

        g = 9.8     #Gravitational acceleration
        h = 7.0       #Temperature gradient parameter
        c = 0.2     #Coriolis effect paramter

        sol_x = np.zeros((nt, self.ny, self.nx))
        sol_x[0] = 0.0
        sol_y = np.zeros((nt, self.ny, self.nx))
        sol_y[0] = 0.0
        
        #get inverted latitude map for coriolis effect
        latitude = (self.lat_map - 1) * (-1)
        self.recalc_temperature(0)
        for t in range(1,nt,1):
            v_arr = np.zeros((self.ny, self.nx))
            u_arr = np.zeros((self.ny, self.nx))
            #Recalculate temperature if needed
            t_m = t * dt
            if float(t_m).is_integer():
                self.recalc_temperature(t_m)
            """
            nx = self.nx
            ny = self.ny
            slopes = self.slopes
            temp_grad = self.temp_grad
            step = timestep_wind(nx, ny,dt, t, sol_x, sol_y, slopes, temp_grad, latitude, [g,h,c,r1,r2,r3,r4])
            u_arr = step[0]
            v_arr = step[1]
            """
            for i in range(0,self.ny,1):
                v_row = np.zeros(self.nx)
                u_row = np.zeros(self.nx)
                for j in range(0,self.nx,1):
                    if j == 0:
                        u = 0
                        v = 0
                    elif j == self.nx - 1:
                        u = 0
                        v = 0
                    elif i == 0:
                        u = 0
                        v = 0
                    elif i == self.ny - 1:
                        u = 0
                        v = 0
                    else:             
                        u = sol_x[t - 1][i][j] - r1 * sol_x[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i - 1][j])\
                                - r2 * sol_y[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i][j - 1])\
                                + r3 * (sol_x[t - 1][i + 1][j] + sol_x[t - 1][i - 1][j] - 2 * sol_x[t - 1][i][j])\
                                + r4 * (sol_x[t - 1][i][j + 1] + sol_x[t - 1][i][j - 1] - 2 * sol_x[t - 1][i][j])\
                                - g * self.slopes[0][i][j] * dt + h * self.temp_grad[0][i][j] * dt \
                                    + latitude[i][j] * c * dt
                        v = sol_y[t - 1][i][j] - r2 * sol_y[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i][j - 1])\
                                - r1 * sol_x[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i - 1][j])\
                                + r3 * (sol_y[t - 1][i + 1][j] + sol_y[t - 1][i - 1][j] - 2 * sol_y[t - 1][i][j])\
                                + r4 * (sol_y[t - 1][i][j + 1] + sol_y[t - 1][i][j - 1] - 2 * sol_y[t - 1][i][j])\
                                - g * self.slopes[1][i][j] * dt + h * self.temp_grad[1][i][j] * dt
                    v_row[j] = u
                    u_row[j] = v
                v_arr[i] = v_row
                u_arr[i] = u_row
            
            sol_x[t] = v_arr
            sol_y[t] = u_arr

            print(t)
        return sol_x, sol_y
    
    def gen_temp_map(self, month):
        """
        Temperature depends on elevation and latitude and month of the year
        Temperature cycle over the year is modeled by a sine function
        Total effect of latitude depends on a base latitude map that reduces temperature near the poles
        Two other latitude maps have highest values either at the south or the north pole
        Total effect of latitude is found by weighting north and south latitude effect based on month, then added to the base latitude
        TODO: Implement offset season for northern and southern hemisphere
        """
        temp_map = (self.h_map - 1) * (-1)
        latmap = noisegen.LatMap(self.height, self.length)
        latitude_base = latmap.gen_lat_map()
        latitude_south = latmap.gen_lat_map(symmetric = False)
        latitude_south = latmap.invert_map(latitude_south)      #Invert one latitude map so that the poles will cycle
        latitude_north = latmap.gen_lat_map(symmetric = False)
        
        
        latitude_ns = latitude_north * (month/11) + latitude_south * ((12 - month) / 11)
        latitude = latitude_ns * 0.1 + latitude_base
        latitude = normalize_map(latitude)

        #Make the temperature cycle with the input month
        #period_coeff = math.pi/6       #Period of the cycle is set to 12
        #offset_coeff = 5                #Offset of sine function set to 4 to make the min value equal to 12
        #latitude = (0.15 * np.sin(latitude + offset_coeff + period_coeff * month) + 0.8)
        temp_map = (temp_map * latitude)
        
        #Increase temperature over the ocean
        #temp_map = np.where(self.h_map < self.water_level, temp_map + 0.05, temp_map)
        #Normalize
        temp_map = normalize_map(temp_map)
        return temp_map

    def gen_humid_map(self):
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

