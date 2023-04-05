import numpy as np
from numba import njit, float32
import numba

"""
Module with numba enhanced functions for faster climate simulations
"""

@njit
def wind_sim(ny, nx, dt, lat_map_inv, temp_gradient, slopes):
   
    """
    Calculate a wind flow map based on a 2D Burgers equation
    Calculations are based on the height map. A negative slope accelerates wind speed due to gravity (g). Opposite is true for upward facing slopes
    Wind speed and direction also depends on temperature gradient.
    A positive temperature gradient (cold air goes to warm air) increases wind speeds
    A latitude term is added to the x-direction of the simulation to mimic a coriolis type effect
    Equations are:
        dv_dt = -dv_dx * v - dv_dy * u  + v * (d2v/dx2 + d2v/dy) - g * s_x(x,y) + h * tg_x(x,y) + c * latitude(x,y)
        du_dt = -du_dx * v - du_dy * u  + v * (d2v/dx2 + d2v/dy) - g * s_y(x,y) + h * tg_y(x,y)
    Where g is gravitational acceleration, and s_x and s_y are the gradients in the x and y directions. 
    tg is the temperature gradient. c is the coriolis force parameter multiplied by latitude
    The wind is simulated taking temperature as predetermined
    """

    #g = 2.72
    g = 0.34
    c = 0.2
    h = 0.75
    nt = int(24 / dt)    #Each run of the wind simulation is one day, so for dt = 1 a time step is one hour
    ny_nx = ny / nx

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
                    u = sol_x[t - 1][i][nx - 1]
                    v = sol_y[t - 1][i][nx - 1]
                elif i == 0:
                    u = 0.25
                    v = 0.0
                elif i == ny:
                    u = 0.25
                    v = 0.0
                else:             
                    u = sol_x[t - 1][i][j] \
                            - dt * sol_x[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i][j - 1])\
                            - dt * sol_y[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i - 1][j])\
                            - dt * g * slopes[0][i][j] \
                            + dt * h * temp_gradient[0][i][j]\
                            + dt * lat_map_inv[i][j] * c

                    v = sol_y[t - 1][i][j] \
                            - dt * sol_y[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i - 1][j])\
                            - dt * sol_x[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i][j - 1])\
                            - dt * g * slopes[1][i][j]\
                            + dt * h * temp_gradient[1][i][j]

                v_row[j] = v
                u_row[j] = u
            v_arr[i] = v_row
            u_arr[i] = u_row
        sol_x[t] = u_arr
        sol_y[t] = v_arr
        print('Period:' + str(t))


    return sol_x, sol_y

@njit
def wind_heat_sim(ny, nx, dt, lat_map_inv, init_temp, slopes):

    """
    Calculate a wind flow map based on a 2D Burgers equation
    Calculations are based on the height map. A negative slope accelerates wind speed due to gravity (g). Opposite is true for upward facing slopes
    Wind speed and direction also depends on temperature gradient.
    A positive temperature gradient (cold air goes to warm air) increases wind speeds
    A latitude term is added to the x-direction of the wind simulation to mimic a coriolis type effect
    Equations for wind are:
        dv_dt = -dv_dx * v - dv_dy * u - g * s_x(x,y) + c * latitude(x,y)
        du_dt = -du_dx * v - du_dy * u - g * s_y(x,y)
    Where g is gravitational acceleration, and s_x and s_y are the gradients in the x and y directions. 
    c is the coriolis force parameter multiplied by latitude
    This method uses forward finite differences
    """

    nt = int(24 / dt)    #Each run of the wind simulation is one day, so for dt = 1 a time step is one hour
    ny_nx = ny / nx

    #Set parameters

    g = 0.34
    c = 0.1
    f = 0.3

    sol_x = np.zeros((nt, ny, nx))
    sol_x[0] = 0.0
    sol_y = np.zeros((nt, ny, nx))
    sol_y[0] = 0.0
    sol_T = np.zeros((nt, ny, nx))
    sol_T[0] = init_temp

    for t in range(1,nt,1):
        v_arr = np.zeros((ny, nx))
        u_arr = np.zeros((ny, nx))
        T_arr = np.zeros((ny, nx))
        for i in range(0,ny):
            v_row = np.zeros(nx)
            u_row = np.zeros(nx)
            T_row = np.zeros(nx)
            for j in range(0,nx):
                if j == 0:
                    u = sol_x[t - 1][i][nx - 1]
                    v = sol_y[t - 1][i][nx - 1]
                    T = sol_T[t - 1][i][nx - 1]
                elif i == 0:
                    u = 0.25
                    v = 0.0
                    T = 0.0
                elif i == ny:
                    u = 0.25
                    v = 0.0
                    T = 0.0
                else:

                    u = sol_x[t - 1][i][j] \
                            - dt * sol_x[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i][j - 1]) \
                            - dt * sol_y[t - 1][i][j] * (sol_x[t - 1][i][j] - sol_x[t - 1][i - 1][j]) \
                            - dt * g * slopes[0][i][j]\
                            + dt * f * (sol_T[t - 1][i][j] - sol_T[t - 1][i][j - 1]) \
                            + dt * lat_map_inv[i][j] * c
                    
                    v = sol_y[t - 1][i][j] \
                            - dt * sol_x[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i][j - 1]) \
                            - dt * sol_y[t - 1][i][j] * (sol_y[t - 1][i][j] - sol_y[t - 1][i - 1][j]) \
                            - dt * g * slopes[1][i][j]\
                            + dt * f * (sol_T[t - 1][i][j] - sol_T[t - 1][i - 1][j])

                    T = sol_T[t - 1][i][j] \
                            - dt * sol_x[t - 1][i][j] * (sol_T[t - 1][i][j] - sol_T[t - 1][i][j - 1]) \
                            - dt * sol_y[t - 1][i][j] * (sol_T[t - 1][i][j] - sol_T[t - 1][i - 1][j])

                u_row[j] = u
                v_row[j] = v
                T_row[j] = T
            u_arr[i] = u_row
            v_arr[i] = v_row
            T_arr[i] = T_row
        sol_x[t] = u_arr
        sol_y[t] = v_arr
        sol_T[t] = T_arr
        print("Period:" + str(t))

    return sol_x, sol_y, sol_T
