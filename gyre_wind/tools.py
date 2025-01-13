import xarray as xr
import numpy as np
import pandas as pd
import datetime



def convert_longitudes(ds):
    """
    Convert longitudes from 0:360 to -180:180 if they go above 180.
    
    Parameters:
    ds (xarray.Dataset): The dataset with longitudes to convert.
    
    Returns:
    xarray.Dataset: The dataset with converted longitudes.
    """
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    return ds

def create_monthly_means(data):
    """
    Resample the input data to monthly means and shift the time index to the 15th of each month.

    Parameters:
    data (xarray.DataArray or xarray.Dataset): The input data to be resampled. It must have a time dimension.

    Returns:
    xarray.DataArray or xarray.Dataset: The resampled data with monthly means and time index shifted to the 15th of each month.

    Notes:
    Original author: Cristina Arumi Planas (2024)
    """

    # Resample data to monthly means
    monthly_data = data.resample(time='M').mean()
    # Shift the time index to the 15th of each month
    monthly_data['time'] = monthly_data['time'] + np.timedelta64(14, 'D')
    return monthly_data


def compute_wind_stress_curl(u10, v10, rho_air=1.225, Cd=1.3e-3, R=6371000):
    """
    Compute the wind stress curl from u and v wind components.
    
    Parameters:
    u10 (xarray.DataArray): Zonal wind component.
    v10 (xarray.DataArray): Meridional wind component.
    rho_air (float): Air density in kg/m^3. Default is 1.225.
    Cd (float): Drag coefficient. Default is 1.3e-3.
    R (float): Earth's radius in meters. Default is 6371000.
    
    Returns:
    xarray.DataArray: Wind stress curl in N/m^3 (multiplied by 1e9).

    Notes:
    Original author: Cristina Arumi Planas (2024)
    """
    # Wind speed in m/s
    wind_speed = np.sqrt(u10**2 + v10**2)

    # wind stress components in N/m^2
    tau_x = rho_air * Cd * u10 * wind_speed
    tau_y = rho_air * Cd * v10 * wind_speed

    # Derivatives
    # Note from Eleanor: Does this take into account the different km-grid spacing in the x and y directions?
    d_tau_y_dx = tau_y.differentiate('longitude')
    d_tau_x_dy = tau_x.differentiate('latitude')

    # Wind stress curl calculation N m-3
    curl_tau = (d_tau_y_dx - d_tau_x_dy) / (R * np.cos(np.deg2rad(u10.latitude)))
    return curl_tau * 1e9

def compute_Sverdrup_transport(curl_tau, latitude, rho0=1030, omega=7.2921e-5, R=6371000):
    """
    Compute the Sverdrup transport, Ekman contribution, and geostrophic contribution.

    Parameters:
    curl_tau (array-like): Wind stress curl (N/m^3) averaged in latitude
    latitude (array-like): Latitude values (degrees).
    rho0 (float, optional): Reference density of seawater (kg/m^3). Default is 1030.
    omega (float, optional): Angular velocity of the Earth (rad/s). Default is 7.2921e-5.
    R (float, optional): Radius of the Earth (m). Default is 6371000.

    Returns:
    tuple: A tuple containing:
        - V_sv (array-like): Sverdrup transport (m^2/s).
        - V_ek (array-like): Ekman contribution to Sverdrup transport (m^2/s).
        - V_g (array-like): Geostrophic contribution to Sverdrup transport (m^2/s).

    Notes:
    Original author: Cristina Arumi Planas (2024)
    """
    f = 2 * omega * np.sin(np.deg2rad(latitude))  # Coriolis parameter (f)
    beta = 2 * omega / R * np.cos(np.deg2rad(latitude))  # Meridional Derivative of the Coriolis Parameter (beta)
    f_lat_avg = f.mean(dim='latitude')
    beta_lat_avg = beta.mean(dim='latitude')

    V_sv = (curl_tau / (beta_lat_avg * rho0))  # Sverdrup transport
    V_ek = (-curl_tau / (rho0 * f_lat_avg))  # Ekman contribution to Sverdrup Transport
    V_g = (V_sv / 1e9 - V_ek / 1e3)  # Geostrophic contribution to Sverdrup Transport

    return V_sv, V_ek, V_g

# Compute and remove seasonal cycle
def remove_seasonal_cycle(region):
    # Group by month and compute the climatological mean for each month
    climatology = region.groupby('time.month').mean(dim='time')
    # Group by month again and subtract the climatology to remove the seasonal cycle
    deseasonalized = region.groupby('time.month') - climatology
    return deseasonalized


def compute_seasonal_cycle(region):
    # Group by month and compute the climatological mean and std for each month
    climatology = region.groupby('time.month').mean(dim='time')
    climatology_std = region.groupby('time.month').std(dim='time')
    
    return climatology, climatology_std


# Apply a rolling boxcar filter (3-month window)
def apply_boxcar_filter(data, window_size):
    return data.rolling(time=window_size, center=True).mean()

# Apply a combined 3-month and 1-year boxcar filter
def apply_combined_boxcar_filter(data):
    data_3mo = apply_boxcar_filter(data, 3)
    data_1yr = apply_boxcar_filter(data_3mo, 12)
    return data_1yr

def process_fc_data(ds_fc_full):
        # Convert to DataFrame
        df = ds_fc_full.to_dataframe()

        # Calculate monthly means
        monthly_data = df.groupby([df.index.year, df.index.month])

        # Calculate the monthly mean, but only if the number of valid days (non-NaN) is at least 15 days
        monthly_mean = monthly_data['Transport'].agg(lambda x: np.nan if len(x) <= 15 else np.mean(x))
        monthly_mean.index = pd.to_datetime([f'{int(year)}-{int(month):02d}-01' for year, month in monthly_mean.index])

        # Fill missing months with NaN
        monthly_mean = monthly_mean.resample('M').mean()

        # Find gaps
        gaps = monthly_mean.isna()

        # Detect gaps smaller than or equal to 3 months, for that we will use rolling
        small_gaps = gaps.rolling(window=4, min_periods=1).sum() <= 3

        # Interpolate only for gaps of 3 months or fewer
        monthly_mean_interpolated = monthly_mean.copy()
        monthly_mean_interpolated[small_gaps] = monthly_mean_interpolated.interpolate()

        # Find gaps (NaN values) in the interpolated data
        gaps_interpolated = monthly_mean_interpolated.isna()

        # Filter the months (index) where the values are NaN
        missing_months = monthly_mean_interpolated.index[gaps_interpolated]

        # Print the missing months and the total number of missing values
        print(f"Missing values found in {len(missing_months)} months.")
        print("Months with missing values:")
        for month in missing_months:
            print(month)

        # Convert back to xarray Dataset
        ds_monthly_mean_interpolated = monthly_mean_interpolated.to_xarray()

        return ds_monthly_mean_interpolated