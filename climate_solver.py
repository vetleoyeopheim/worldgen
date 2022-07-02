import numpy as np
from numba import njit, float32
import numba

"""
Module with numba enhanced functions for faster climate simulations
"""

@njit
def windheat_sim(nx, ny, nt, dt, slopes, surf_slopes, latitude, temp_map, height_map,  params, land_map):
   
    """
    Calculate a wind flow map based on a 2D Burgers equation

    Calculations are based on the height map. A negative slope accelerates wind speed due to gravity (g). Opposite is true for upward facing slopes

    Wind speed and direction also depends on temperature gradient.

    A positive temperature gradient (cold air goes to warm air) increases wind speeds

    A latitude term is added to the x-direction of the wind simulation to mimic a coriolis type effect

    Temperature also changes with the time of the year using a sine function

    Equations for wind are:
        dv_dt = -dv_dx * v - dv_dy * u  + v * (d2v/dx2 + d2v/dy) - g * s_x(x,y) + h * tg_x(x,y) + c * latitude(x,y)
        du_dt = -du_dx * v - du_dy * u  + v * (d2v/dx2 + d2v/dy) - g * s_y(x,y) + h * tg_y(x,y)
    Where g is gravitational acceleration, and s_x and s_y are the gradients in the x and y directions. 
    tg is the temperature gradient. c is the coriolis force parameter multiplied by latitude
    Temperature affects wind dynamically
    This method uses forward finite differences
    """
    g = params[0]
    h = params[1]
    c = params[2]
    
    r1 = params[3]
    r2 = params[4]
    r3 = params[5]
    r4 = params[6]

    alpha = params[7]

    sol_x = np.zeros((nt, ny, nx))
    sol_x[0] = 0.0
    sol_y = np.zeros((nt, ny, nx))
    sol_y[0] = 0.0
    sol = np.zeros((2,nt,ny,nx))
    temp = np.zeros((nt, ny, nx))
    temp[0] = temp_map
    inv_lat = (latitude - 1) * (-1)         #Invert latitude map for temperature calculation
    
    for t in range(1,nt,1):
        v_arr = np.zeros((ny, nx))
        u_arr = np.zeros((ny, nx))
        T_arr = np.zeros((ny, nx))
        for i in range(0,ny,1):
            v_row = np.zeros(nx)
            u_row = np.zeros(nx)
            T_row = np.zeros(nx)
            for j in range(0,nx,1):
                if j == 0:
                    u = 0.0
                    v = 0.0
                    T = temp[0][i][j]
                elif j == nx - 1:
                    u = 0.0
                    v = 0.0
                    T = temp[0][i][j]
                elif i == 0:
                    u = 0.0
                    v = 0.0
                    T = temp[0][i][j]
                elif i == ny - 1:
                    u = 0.0
                    v = 0.0
                    T = temp[0][i][j]
                else:
                    u = sol_x[t - 1][i][j] - r1 * sol_x[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i - 1][j])\
                            - r2 * sol_y[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i][j - 1])\
                            + r3 * (sol_x[t - 1][i + 1][j] + sol_x[t - 1][i - 1][j] - 2 * sol_x[t - 1][i][j])\
                            + r4 * (sol_x[t - 1][i][j + 1] + sol_x[t - 1][i][j - 1] - 2 * sol_x[t - 1][i][j])\
                            - g * slopes[0][i][j] * dt\
                            + h * (temp[t - 1][i][j] - temp[t - 1][i - 1][j]) * dt\
                            + inv_lat[i][j] * c * dt\
                            - 0.0 * land_map[i][j] * dt

                    v = sol_y[t - 1][i][j] - r2 * sol_y[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i][j - 1])\
                            - r1 * sol_x[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i - 1][j])\
                            + r3 * (sol_y[t - 1][i + 1][j] + sol_y[t - 1][i - 1][j] - 2 * sol_y[t - 1][i][j])\
                            + r4 * (sol_y[t - 1][i][j + 1] + sol_y[t - 1][i][j - 1] - 2 * sol_y[t - 1][i][j])\
                            - g * slopes[1][i][j] * dt\
                            + h * (temp[t - 1][i][j] - temp[t - 1][i][j - 1]) * dt\
                            - 0.0 * land_map[i][j] * dt

                    T = temp[t - 1][i][j] - dt * sol_x[t - 1][i][j] * (temp[t - 1][i][j] - temp[t - 1][i - 1][j])\
                            - dt * sol_y[t - 1][i][j] * (temp[t - 1][i][j] - temp[t - 1][i][j - 1])\
                            + dt * alpha * (sol_y[t - 1][i + 1][j] + sol_y[t - 1][i - 1][j] - 2 * sol_y[t - 1][i][j])\
                            + dt * alpha * (sol_y[t - 1][i][j + 1] + sol_y[t - 1][i][j - 1] - 2 * sol_y[t - 1][i][j])
                            #+ dt * np.sin((0.5 * 3.14 * t) / nt) * temp[t - 1][i][j] * inv_lat[i][j]
                            #- dt * 0.5 * (temp[t - 1][i][j] - temp[0][i][j])\

                v_row[j] = u
                u_row[j] = v
                T_row[j] = T
            v_arr[i] = v_row
            u_arr[i] = u_row
            T_arr[i] = T_row
        sol_x[t] = v_arr
        sol_y[t] = u_arr
        temp[t] = T_arr
        print(t)
    sol[0] = sol_x
    sol[1] = sol_y

    return sol_x, sol_y, temp

@njit
def humidity_sim(wind_x, wind_y, nx, ny, nt, dt, init_evap,h_map, temps, slopes, params):
    """
    Humidity depends on wind direction, wind strength and evaporation
    Strong winds carry humidity away while evaporation increases humidity on a given point
    Humidity also drops as elevation increases
    Solution technique is a forward finite differences method
    Evaporation is recalculated based on humidity level
    """
    k = params[0]     #Diffusion parameter
    p = params[1] 
    d = params[2]     #Elevation parameter
    e = params[3]     #Evaporation parameter
    h_l_ratio = params[4]
    T = params[5]
     
    
    humidity = np.zeros((nt,ny,nx))
    humidity[0] = init_evap    #Initial condiation is just the evaporation or 0
    evaporation = np.zeros((nt,ny,nx))
    evaporation[0] = init_evap
    precipitation = np.zeros((nt,ny,nx))

    for t in range(1,nt,1):
        xy_arr = np.zeros((ny, nx))
        t_index = int(float(t) * dt) - 1
        #Reset t_index to zero if it is greater than 120 (one year has passed)
        t_index = t_index % 12
        print(t_index)
        for i in range(0,ny,1):
            x_col = np.zeros(nx)
            for j in range(0,nx,1):
                if i == ny - 1:
                    h = 0.1
                elif i == 0:
                    h = 0.1
                elif j == 0:
                    h = 0.1
                elif j == nx - 1:
                    h = 0.1
                else:
                    h = humidity[t - 1][i][j]\
                        - dt * p * wind_x[t - 1][i - 1][j] * (humidity[t - 1][i][j] - humidity[t - 1][i - 1][j])\
                        - dt * p * wind_y[t - 1][i][j - 1] * (humidity[t - 1][i][j] - humidity[t - 1][i][j - 1])\
                        + dt * k * (humidity[t - 1][i + 1][j] + humidity[t - 1][i - 1][j] - 2 * humidity[t - 1][i][j])\
                        + dt * k * (humidity[t - 1][i][j + 1] + humidity[t - 1][i][j - 1] - 2 * humidity[t - 1][i][j])\
                        + dt * e * evaporation[t - 1][i][j]\
                        - dt * d * np.exp(h_map[i][j])\
                        - dt * T * np.exp(-temps[t_index][i][j])\

                x_col[j] = h
            xy_arr[i] = x_col
        humidity[t] = xy_arr
        #Recalculate evaporation based on humidity
        evaporation[t] = humidity[t] * 0.5
        #precipitation[t] = (humidity[t])**(2) + (temps[t])**(2)
        print(t)

    return humidity, precipitation

