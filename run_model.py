import xarray as xr
import numpy as np
from scipy import interpolate

# Project libs
from shallow_water import config  # File with integration parameters
from shallow_water.model import shallow_water


def init_C_grid(Lx, Ly, dx, dy, time_vector):
    ntime = len(time_vector)
    # Initializing coordinates with one extra zonal grid point in u and one extra meridional grid point in v
    u_coords = {'x': np.arange(-0.5 * dx, Lx + 0.5 * dx, dx), 'y': np.arange(-Ly/2, Ly/2, dy)}
    v_coords = {'x': np.arange(0, Lx, dx), 'y': np.arange(-(Ly/2) -0.5 * dy, (Ly/2) + 0.5 * dy, dy)}
    eta_coords = {'x': np.arange(0, Lx, dx), 'y': np.arange(-(Ly/2), Ly/2, dy)}
    nx = config.nx
    ny = config.ny
    # Initializing 2D xarray dataarrays for storing a single time step (used during integration)
    eta_array = xr.DataArray(np.zeros([ny, nx]), dims=('y', 'x'), coords={'y': eta_coords['y'], 'x': eta_coords['x']})
    u_array = xr.DataArray(np.zeros([ny, nx + 1]), dims=('y', 'x'), coords={'y': u_coords['y'], 'x': u_coords['x']})
    v_array = xr.DataArray(np.zeros([ny + 1, nx]), dims=('y', 'x'), coords={'y': v_coords['y'], 'x': v_coords['x']})
    sigma, mu = .5, 0.0
    f = lambda x: np.exp(-((x - mu) ** 2 / (2.0 * sigma ** 2)))
    x, y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    d = np.sqrt(x * x + y * y)
    eta_array = xr.DataArray(f(d)/10, dims=('y', 'x'), coords={'y': eta_coords['y'], 'x': eta_coords['x']})


    # Initializing 3D dataarrays with a time dimension to keep record of the time steps
    eta_array_dump = xr.DataArray(np.zeros([ny, nx, ntime]), dims=('y', 'x', 'time'),
                                  coords={'y': eta_coords['y'], 'x': eta_coords['x'], 'time': time_vector})
    u_array_dump = xr.DataArray(np.zeros([ny, nx + 1, ntime]), dims=('y', 'x', 'time'),
                                coords={'y': u_coords['y'], 'x': u_coords['x'], 'time': time_vector})
    v_array_dump = xr.DataArray(np.zeros([ny + 1, nx, ntime]), dims=('y', 'x', 'time'),
                                coords={'y': v_coords['y'], 'x': v_coords['x'], 'time': time_vector})
    return eta_array, u_array, v_array, eta_array_dump, u_array_dump, v_array_dump


def divergence(u, v, dx, dy):
    dudx = u.diff('x') / dx
    dvdy = v.diff('y') / dy
    dudx['x'] = dvdy.x  # Fixing the coordinates to match eta
    dvdy['y'] = dudx.y
    return dudx + dvdy


def zonal_boundary(u):
    u[:, [0, -1]] = 0
    return u


def meridional_boundary(v):
    v[[0, -1], :] = v[[1, -2], :].values
    v[[0, -1], :] = 0
    return v


def coriolis(beta_plane, u, v, component):
    '''
    Function to compute the Coriolis term in a beta-plane approximation.

    Here we use the interp method from the xarray library (same as numpy interp)
    to compute the v values in the u grid (and vice-versa)
    '''
    if component == 'zonal':
        coriolis_force = beta_plane.interp(y=u.y, method='linear') * v[:, :].interp(x=u.x[1:-1], y=u.y, method='linear')
    elif component == 'meridional':
        coriolis_force = beta_plane.interp(y=v.y, method='linear') * u[:, :].interp(x=v.x, y=v.y[1:-1], method='linear')
    else:
        raise ValueError(f'Component {component} not available')
    return coriolis_force


def interp_at_depart(A, u, v, dt):
    '''
    Method for finding the departure points of the quantity A (xarray dataarray)
    '''
    # Finding departure points:
    x_dep = A.x - dt * u.interp(x=A.x, y=A.y)
    y_dep = A.y - dt * v.interp(x=A.x, y=A.y)

    # Boundary conditions: truncating the departure points at the border of the domain
    x_dep = x_dep.where(x_dep >= A.x.min(), A.x.min())
    x_dep = x_dep.where(x_dep <= A.x.max(), A.x.max())
    y_dep = y_dep.where(y_dep >= A.y.min(), A.y.min())
    y_dep = y_dep.where(y_dep <= A.y.max(), A.y.max())

    # Interpolating A at the departure points
    A.values = A.interp(x=x_dep, y=y_dep, method='linear').values

    return A


def second_order_interp(A, u, v, dt):
    '''
    Method to obtain a fully second order semi-lagrangian scheme
    followin Durran's suggestions
    '''
    # Step1: Finding the midpoint of the back trajectory
    x_mid = A.x - 0.5 * dt * u.interp(x=A.x, y=A.y)
    y_mid = A.y - 0.5 * dt * v.interp(x=A.x, y=A.y)

    # Truncating the midpoints points at the border of the domain
    x_mid = x_mid.where(x_mid >= A.x.min(), A.x.min())
    x_mid = x_mid.where(x_mid <= A.x.max(), A.x.max())
    y_mid = y_mid.where(y_mid >= A.y.min(), A.y.min())
    y_mid = y_mid.where(y_mid <= A.y.max(), A.y.max())

    # Step 2: computing u and v by linear interpolation in the midpoints
    u_mid = u.interp(x=x_mid.x, y=y_mid.y, method='linear')
    v_mid = v.interp(x=x_mid.x, y=y_mid.y, method='linear')

    # Step 3: determining the departure point
    x_dep = A.x - dt * u_mid
    y_dep = A.y - dt * v_mid

    # Step 4: performing a quadratic interpolation
    A_df = A.to_dataframe
    interp_f = interpolate.interp2d(A.x.values, A.y.values, A.values, kind='cubic')
    A.values = interp_f(x_dep.x.values, y_dep.y.values)
    return (A)


def run_integration(sw_model):
    '''
    Function to integrate the shallow water model in the Arakawa C-grid.
    Please, refer to config.py and model.py for more information about the model parameters.
    '''

    Lx = config.Lx
    Ly = config.Ly
    dx = config.dx
    dy = config.dy
    method = config.method

    if config.delta_t == 'Maximum stable':
        dt = cfl_2D(min(dx, dy), sw_model)
        print("Using dt " + str(dt))
    else:
        dt = config.delta_t

    ntime = int(config.integration_length * 86400 / dt)

    time_vector = np.arange(0, ntime * dt, dt)
    eta, u, v, eta_dump, u_dump, v_dump = init_C_grid(Lx, Ly, dx, dy, time_vector)
    tau_x, tau_y = sw_model.calc_tau(u.y, Ly)
    beta_plane = (sw_model.f_0 + sw_model.beta * eta.y)

    if method == 'eulerian':  # Just defining an identity function for the eulerian version
        f = lambda x, u, v, t: x
    elif method == 'semi lagrangian':
        f = interp_at_depart
    elif method == '2nd ord. SL':
        f = second_order_interp

    print("Starting temporal loop")
    for t in np.arange(0, ntime - 2,
                       2):  # I found that copying the equations in reversed order is faster than using an if statement

        eta[:, :] = f(eta[:, :], u[:, :], v[:, :], dt) - sw_model.H * dt * divergence(u, v, dx, dy)

        height_div = eta[:, :].diff('x') / dx
        height_div['x'] = u.x.values[
                          1:-1]  # this is a silly coordinate fix because .diff doesn't return the midpoint coordinate
        u[:, 1:-1] = f(u[:, 1:-1], u[:, :], v[:, :], dt) + dt * coriolis(beta_plane, u, v, 'zonal') - \
                     sw_model.g * dt * height_div - sw_model.gamma * dt * u[:, 1:-1] + \
                     tau_x / (sw_model.rho * sw_model.H) * dt
        u = zonal_boundary(u)

        height_div = eta[:, :].diff('y') / dy
        height_div['y'] = v.y.values[1:-1]

        v[1:-1, :] = f(v[1:-1, :], u[:, :], v[:, :], dt) - dt * coriolis(beta_plane, u, v, 'meridional') - \
                     sw_model.g * dt * height_div - sw_model.gamma * dt * v[1:-1, :] + \
                     tau_y / (sw_model.rho * sw_model.H) * dt
        v = meridional_boundary(v)

        u_dump[:, :, t] = u
        v_dump[:, :, t] = v
        eta_dump[:, :, t] = eta

        t += 1  # Updating t and reversing order of u and v

        eta[:, :] = f(eta[:, :], u[:, :], v[:, :], dt) - sw_model.H * dt * divergence(u, v, dx, dy)

        height_div = eta[:, :].diff('y') / dy
        height_div['y'] = v[1:-1, :].y
        v[1:-1, :] = f(v[1:-1, :], u[:, :], v[:, :], dt) - dt * coriolis(beta_plane, u, v, 'meridional') - \
                     sw_model.g * dt * height_div - sw_model.gamma * dt * v[1:-1, :] + \
                     tau_y / (sw_model.rho * sw_model.H) * dt
        v = meridional_boundary(v)

        height_div = eta[:, :].diff('x') / dx
        height_div['x'] = u[:, 1:-1].x
        u[:, 1:-1] = f(u[:, 1:-1], u[:, :], v[:, :], dt) + dt * coriolis(beta_plane, u, v, 'zonal') - \
                     sw_model.g * dt * height_div - sw_model.gamma * dt * u[:, 1:-1] + \
                     tau_x / (sw_model.rho * sw_model.H) * dt
        u = zonal_boundary(u)

        u_dump[:, :, t] = u
        v_dump[:, :, t] = v
        eta_dump[:, :, t] = eta
        print('Integrating day ' + str(1 + int(t * dt / 86400.)))


    if config.output_netcdf:  # dump as netcdf in the output folder (may take a while for long runs)
        eta_dump.to_netcdf('outputs/eta_{method}_resolution_{dx}km.nc'.format(method=method, dx=dx / 1000))
        u_dump.to_netcdf('outputs/u_{method}_resolution_{dx}km.nc'.format(method=method, dx=dx / 1000))
        v_dump.to_netcdf('outputs/v_{method}_resolution_{dx}km.nc'.format(method=method, dx=dx / 1000))

    return eta_dump, u_dump, v_dump


def cfl_2D(d, sw_model):
    '''
    Function to return a stable time-step for a given grid spacing
    respecting the 2D cfl criterea
    '''
    delta_t = 0.5 * d * 2 ** (-0.5) * (sw_model.g * sw_model.H) ** (-0.5)
    return delta_t


def run_analytical_solution(sw_model):
    Lx = config.Lx
    Ly = config.Ly
    dx = config.dx
    dy = config.dy
    nx = config.nx
    ny = config.ny

    x_coords = np.arange(0, Lx, dx)
    y_coords = np.arange(0, Ly, dy)

    u_array = xr.DataArray(np.zeros([ny, nx]), dims=('y', 'x'))
    v_array = xr.DataArray(np.zeros([ny, nx]), dims=('y', 'x'))
    eta_array = xr.DataArray(np.zeros([ny, nx]), dims=('y', 'x'))

    u_array['x'], u_array['y'] = x_coords, y_coords
    v_array['x'], v_array['y'] = x_coords, y_coords
    eta_array['x'], eta_array['y'] = x_coords, y_coords

    # The method "apply_ufunc" applies the required function (sw_model.analytical_solution) in each cell of the array.
    # For documentation please visit http://xarray.pydata.org/en/stable/generated/xarray.apply_ufunc.html
    u_sol = xr.apply_ufunc(lambda u, v, eta: sw_model.analytical_solution(u, v, eta)[0], u_array.x, v_array.y,
                           eta_array)
    v_sol = xr.apply_ufunc(lambda u, v, eta: sw_model.analytical_solution(u, v, eta)[1], u_array.x, v_array.y,
                           eta_array)
    eta_sol = xr.apply_ufunc(lambda u, v, eta: sw_model.analytical_solution(u, v, eta)[2], u_array.x, v_array.y,
                             eta_array)

    # plot_analytical(u_sol, v_sol, eta_sol)
    return u_sol, v_sol, eta_sol


def calc_energy(u, v, eta, sw_model):
    u = u.interp(x=eta.x)
    v = v.interp(y=eta.y)
    E = 0.5 * sw_model.rho * (sw_model.H * (u ** 2 + v ** 2) + sw_model.g * eta ** 2)
    return E


if __name__ == '__main__':
    '''
    Please see the config file before running the main function.

    The netcdfs and figures in the folder will be overwritten every time you run the model
    '''
    sw_model = shallow_water(config.Lx, config.Ly, f_0=0)  # Create instance of sw model with default parameters
    eta, u, v = run_integration(sw_model)
