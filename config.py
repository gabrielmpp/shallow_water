'''
All relevant parameters for running the model can be controlled from this file
The shallow water parameters are in the model.py

Note: The code was written using Python 3.
'''

import os

# Get project basepath
base_path = os.path.dirname(os.path.abspath(__file__))
# Length of the domain in meters
L = 1e6
# Number of grid points in x and y
nx = 20
ny = 20
# Grid spacing
dx = L/nx
dy = L/ny
# Duration of the integration in days (to reproduce task D set as 1)
integration_length = 1
# Time step size. Set for Maximum stable to use a large but safe time step.
delta_t = 'Maximum stable'
# Possiblie options for method are: 'eulerian', 'semi lagrangian' and '2nd ord. SL'
method = "eulerian"
# Should the output file be saved in netcdf? The saving process may take a while
output_netcdf = True
