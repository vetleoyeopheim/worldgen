import numpy as np
from numba import njit, float32
import numba

"""
Module with numba enhanced functions for faster climate simulations
"""

@njit(parallel = False)
def wind_sim(nx, ny, nt, dt, slopes, temp_grad, latitude, params):
   
    """
    Calculate a wind flow map based on a 2D Burgers equation
    Calculations are based on the height map. A negative slope accelerates wind speed due to gravity (g). Opposite is true for upward facing slopes
    Wind speed and direction also depends on temperature gradient.
    A positive temperature gradient (cold air goes to warm air) increases wind speeds
    A latitude term is added to the x-direction of the simulation to mimic a coriolis type effect
    Equations are:
        dv_dx = -du_dy - v * (d2v/dx2 + d2v/dy) - g * s_x(x,y) + h * tg_x(x,y) + c * latitude(x,y)
        du_dy = -dv_dx - v * (d2u/dx2 + d2u/dy) - g * s_y(x,y) + h * tg_y(x,y)
    Where g is gravitational acceleration, and s_x and s_y are the gradients in the x and y directions. 
    tg is the temperature gradient
    """
    g = params[0]
    h = params[1]
    c = params[2]
    
    r1 = params[3]
    r2 = params[4]
    r3 = params[5]
    r4 = params[6]

    v = params[7]

    sol_x = np.zeros((nt, ny, nx))
    sol_x[0] = 0.0
    sol_y = np.zeros((nt, ny, nx))
    sol_y[0] = 0.0
    sol = np.zeros((2,nt,ny,nx))

        
    for t in range(1,nt,1):
        v_arr = np.zeros((ny, nx))
        u_arr = np.zeros((ny, nx))
        for i in range(0,ny,1):
            v_row = np.zeros(nx)
            u_row = np.zeros(nx)
            for j in range(0,nx,1):
                if j == 0:
                    u = 0.0
                    v = 0.0
                elif j == nx - 1:
                    u = 0.0
                    v = 0.0
                elif i == 0:
                    u = 0.0
                    v = 0.0
                elif i == ny - 1:
                    u = 0.0
                    v = 0.0
                else:             
                    u = sol_x[t - 1][i][j] - r1 * sol_x[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i - 1][j])\
                            - r2 * sol_y[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i][j - 1])\
                            + r3 * (sol_x[t - 1][i + 1][j] + sol_x[t - 1][i - 1][j] - 2 * sol_x[t - 1][i][j])\
                            + r4 * (sol_x[t - 1][i][j + 1] + sol_x[t - 1][i][j - 1] - 2 * sol_x[t - 1][i][j])\
                            - g * slopes[0][i][j] * dt + h * temp_grad[0][i][j] * dt \
                                + latitude[i][j] * c * dt
                    v = sol_y[t - 1][i][j] - r2 * sol_y[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i][j - 1])\
                            - r1 * sol_x[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i - 1][j])\
                            + r3 * (sol_y[t - 1][i + 1][j] + sol_y[t - 1][i - 1][j] - 2 * sol_y[t - 1][i][j])\
                            + r4 * (sol_y[t - 1][i][j + 1] + sol_y[t - 1][i][j - 1] - 2 * sol_y[t - 1][i][j])\
                            - g * slopes[1][i][j] * dt + h * temp_grad[1][i][j] * dt

                v_row[j] = u
                u_row[j] = v
            v_arr[i] = v_row
            u_arr[i] = u_row
        sol_x[t] = v_arr
        sol_y[t] = u_arr
        print(t)
    sol[0] = sol_x
    sol[1] = sol_y

    return sol_x, sol_y

@njit(parallel = False)
def humidity_sim(wind_x, wind_y, nx, ny, nt, dt, evap_map, h_map, params):
    """
    NOTE: THIS RUNS THE HUMIDITY SIMULATION WITHOUT NUMBA, AND IS SIGNIFICANTLY SLOWER. USE THE "climate_solver" version instead
    Humidity depends on wind direction, wind strength and evaporation
    Strong winds carry humidity away while evaporation increases humidity on a given point
    Humidity also drops as elevation increases
    """
    k = params[0]     #Diffusion parameter
    p = params[1] 
    d = params[2]     #Elevation parameter
    e = params[3]     #Evaporation parameter
    h_l_ratio = params[4] 
    
    humidity = np.zeros((nt,ny,nx))
    humidity[0] = evap_map     #Initial condiation is just the evaporation
    for t in range(1,nt,1):
        xy_arr = np.zeros((ny, nx))
        for i in range(0,ny,1):
            x_col = np.zeros(nx)
            for j in range(0,nx,1):
                if i == ny - 1:
                    h = 0
                elif i == 0:
                    h = 0
                elif j == 0:
                    h = 0
                elif j == nx - 1:
                    h = 0
                else:
                    h = humidity[t - 1][i][j]\
                        - dt * p * h_l_ratio * wind_x[t - 1][i][j] * (humidity[t - 1][i][j] - humidity[t - 1][i - 1][j])\
                        - dt * p * wind_y[t - 1][i][j] * (humidity[t - 1][i][j] - humidity[t - 1][i][j - 1])\
                        + dt * k * (h_l_ratio **2) * (humidity[t - 1][i + 1][j] + humidity[t - 1][i - 1][j] - 2 * humidity[t - 1][i][j])\
                        + dt * k * (humidity[t - 1][i][j + 1] + humidity[t - 1][i][j - 1] - 2 * humidity[t - 1][i][j])\
                        + dt * e * evap_map[i][j]\
                        - dt * d * h_map[i][j]
                   
                x_col[j] = h
            xy_arr[i] = x_col
        humidity[t] = xy_arr
        print(t)

    return humidity

