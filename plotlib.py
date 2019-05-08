from model import shallow_water
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import config # File with integration parameters
import os

def plot_analytical(u_sol,v_sol, eta_sol,figname = 'analytical_solution'):
    '''
    This function plots the analytical solution
    '''
    fig = plt.figure()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    eta_sol = eta_sol.transpose('y','x')
    u_sol = u_sol.transpose('y','x')
    v_sol = v_sol.transpose('y','x')
    eta_sol.plot(cmap='RdBu', center=0)
    magnitude = (u_sol**2 + v_sol**2)**0.5
    lw = 3*magnitude/magnitude.max()
    plt.streamplot(u_sol.x.values, v_sol.y.values, u_sol.values, v_sol.values, linewidth = lw.values, color='black')
    plt.savefig(config.base_path + '/figs/task_C/' + figname + '.png')
    plt.close()

def plot_energy(figpath='/figs/task_E/', outpath = '/outputs/'):
    '''
    This function plots all the energy errors available in the outputs path
    as timeseries
    '''
    files_path = config.base_path + outpath
    files = []
    # Loop to fetch all energy error files paths
    for r, d, f in os.walk(files_path):
        for file in f:
            if 'energy_error' in file:
                files.append(os.path.join(r, file))
    legend_labels = []
    plt.figure(figsize=[6,6])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    for file in files:
        print(file)
        E_error = xr.open_dataarray(file)
        E_error_integral =  E_error[:,:,:-2].groupby('time').mean()  # Since the mesh is equally space it is fine to take the mean
        plt.style.use('bmh')
        E_error_integral.plot()
        plt.xlabel('Time (s)')
        plt.ylabel('Energy error')
        legend_labels.append(E_error.name)
    plt.legend(legend_labels)

    plt.savefig(config.base_path + figpath + 'energy_error.pdf')
    plt.close()

def plot_task_D(u, v, eta, figpath = '/figs/task_D/'):
    L = config.L
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Transposing arrays for plot
    eta = eta.transpose('y','x')
    u = u.transpose('y','x')
    v = v.transpose('y','x')

    plt.figure(figsize=[5,5])
    plt.style.use('bmh')
    u.sel(y=0).plot()
    plt.xlabel('u (m/s)')
    plt.ylabel('x (m)')
    plt.savefig(config.base_path + figpath + 'u_vs_x_{method}.pdf'.format(method=config.method))
    plt.close()

    plt.figure(figsize=[5,5])
    plt.style.use('bmh')
    v.sel(x=0).plot()
    plt.ylabel('v (m/s)')
    plt.xlabel('y (m)')
    plt.savefig(config.base_path + figpath + 'v_vs_y_{method}.pdf'.format(method=config.method))
    plt.close()

    plt.figure(figsize=[5,5])
    plt.style.use('bmh')
    eta.sel(y=L/2, method='nearest').plot()
    plt.xlabel('x (m)')
    plt.ylabel('Surface height (m)')
    plt.savefig(config.base_path + figpath + 'eta_vs_x_{method}.pdf'.format(method=config.method))
    plt.close()

    plt.figure(figsize=[6,5])
    eta.plot(cmap='RdBu', cbar_kwargs={'label':'Surface height (m)'})
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    u=u.interp(x=eta.x,y=eta.y,method='linear')
    v=v.interp(x=eta.x,y=eta.y,method='linear')
    magnitude = (u**2 + v**2)**0.5
    lw = 3*magnitude/magnitude.max()
    plt.streamplot(u.x.values, v.y.values, u.values, v.values, linewidth = lw.values, color='black')
    plt.savefig(config.base_path + figpath + 'eta_and_streamlines_{method}.pdf'.format(method=config.method))
    plt.close()

def plot_task_F(figpath='/figs/task_F/'):
     '''
     This function plots the comparision between the eulerian and the SL scheme at 1 day
     This function requires the semi-Lagrangian and eulerian 25km netcdf to be available in the
     outputs dir.
     '''

     try:
         semilagrangian = xr.open_dataarray(config.base_path+'/outputs/eta_semi lagrangian_resolution_25.0km.nc')
         u = xr.open_dataarray(config.base_path+'/outputs/u_semi lagrangian_resolution_25.0km.nc')
         v = xr.open_dataarray(config.base_path+'/outputs/v_semi lagrangian_resolution_25.0km.nc')

         eulerian = xr.open_dataarray(config.base_path+'/outputs/eta_eulerian_resolution_25.0km.nc')

         difference =  semilagrangian - eulerian
         difference = difference.interp(time=86400)
         semilagrangian = semilagrangian.interp(time=86400)
         u = u.interp(time=86400)
         v = v.interp(time=86400)
         plt.figure(figsize=[6,5])
         difference.plot()
         plt.xlabel('x (m)')
         plt.ylabel('y (m)')
         plt.savefig(config.base_path + figpath + 'SL_difference.pdf')
         plt.close()

         plt.figure(figsize=[6,5])
         semilagrangian.plot(cmap='RdBu', cbar_kwargs={'label':'Surface height (m)'})
         plt.xlabel('x (m)')
         plt.ylabel('y (m)')
         u=u.interp(x=semilagrangian.x,y=semilagrangian.y,method='linear')
         v=v.interp(x=semilagrangian.x,y=semilagrangian.y,method='linear')
         magnitude = (u**2 + v**2)**0.5
         lw = 3*magnitude/magnitude.max()
         plt.streamplot(u.x.values, v.y.values, u.values, v.values, linewidth = lw.values, color='black')
         plt.savefig(config.base_path + figpath + 'SL_at_1day.pdf')

     except:
         print("*"*20)
         print("50 Km netcdfs not found in the outputs dir, please run the SL and eulerian schemes with \
                   nx=ny=20 or change the required netcdf file in the 'plot_task_F' funcion in the plotlib.py file. ")
         print("*"*20)

def plot_task_G(eta_s, figpath='/figs/task_G/'):
     '''
     This function requires the 2nd order SL 25km netcdfs to be available in the
     outputs dir. First you need to run the sem lagrangians schemes in 50 km resolution by
     altering the config file.
     '''
     try:
         SL_2nd_order = xr.open_dataarray(config.base_path+'/outputs/eta_2nd ord. SL_resolution_50.0km.nc')

         u = xr.open_dataarray(config.base_path+'/outputs/u_2nd ord. SL_resolution_50.0km.nc')
         v = xr.open_dataarray(config.base_path+'/outputs/v_2nd ord. SL_resolution_50.0km.nc')
         eta_s = eta_s.interp(x=SL_2nd_order.x, y = SL_2nd_order.y)

         u = u.interp(time=86400)
         v = v.interp(time=86400)
         SL_2nd_order = SL_2nd_order.interp(time=86400*29)
         error = SL_2nd_order - eta_s
         plt.figure(figsize=[6,5])
         error.plot()
         plt.xlabel('x (m)')
         plt.ylabel('y (m)')
         plt.savefig(config.base_path + figpath + '2nd_order_SL_error.pdf')
         plt.close()
         plt.figure(figsize=[6,5])
         SL_2nd_order.plot(cmap='RdBu', cbar_kwargs={'label':'Surface height (m)'})
         plt.xlabel('x (m)')
         plt.ylabel('y (m)')
         u=u.interp(x=SL_2nd_order.x,y=SL_2nd_order.y,method='linear')
         v=v.interp(x=SL_2nd_order.x,y=SL_2nd_order.y,method='linear')
         magnitude = (u**2 + v**2)**0.5
         lw = 3*magnitude/magnitude.max()
         plt.streamplot(u.x.values, v.y.values, u.values, v.values, linewidth = lw.values, color='black')
         plt.savefig(config.base_path + figpath + '2nd_ord_SL_at_1day.pdf')
     except:
         print("*"*20)
         print("50 Km netcdfs not found in the outputs dir, please run the SL and eulerian schemes with \
                   nx=ny=20 or change the required netcdf file in the 'plot_task_F' funcion in the plotlib.py file. ")
         print("*"*20)
