import xarray as xr
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd


def load_daily_data(input_dir, u_filename):
    wind10_daily = xr.open_dataset(f'{input_dir}/{u_filename}')
    wind10_daily = convert_longitudes(wind10_daily)
    return wind10_daily

def load_dinezio(path_to_dinezio):
    fname_east = 'Dinezio_Wind_eastern_2_12yr.mat'
    fname_central = 'Dinezio_Wind_central_2_12yr.mat'
    fname_west = 'Dinezio_Wind_western_2_12yr.mat'

    # Load data
    data_east = loadmat(path_to_dinezio + fname_east)
    data_central = loadmat(path_to_dinezio + fname_central)
    data_west = loadmat(path_to_dinezio + fname_west)
    
    # Extract time and wind data
    time_east = data_east['Dinezio_Wind_eastern_2_12yr'][:, 0]
    wind_east = data_east['Dinezio_Wind_eastern_2_12yr'][:, 1]
    
    time_central = data_central['Dinezio_Wind_central_2_12yr'][:, 0]
    wind_central = data_central['Dinezio_Wind_central_2_12yr'][:, 1]
    
    time_west = data_west['Dinezio_Wind_western_2_12yr'][:, 0]
    wind_west = data_west['Dinezio_Wind_western_2_12yr'][:, 1]
    
    # Create xarray datasets
    ds_east = xr.Dataset({'wind': (['time'], wind_east)}, coords={'time': time_east})
    ds_central = xr.Dataset({'wind': (['time'], wind_central)}, coords={'time': time_central})
    ds_west = xr.Dataset({'wind': (['time'], wind_west)}, coords={'time': time_west})
    
    return ds_east, ds_central, ds_west

def load_florida_current_data(input_dir):
    # Load the .mat file
    data = loadmat(os.path.join(input_dir, 'floridacurrent_198204.202012.mat'))
    fc_time = data['fcmonthly']['time'][0,0]
    fc_trans = data['fcmonthly']['trans'][0,0]

    # Flatten arrays if they have more than one dimension
    if fc_time.ndim > 1:
        fc_time = np.ravel(fc_time)
    if fc_trans.ndim > 1:
        fc_trans = np.ravel(fc_trans)

    # Convert numeric time to datetime
    # Assuming fc_time is in months since January 1982
    start_date = pd.Timestamp('1982-01-01')

    # Convert months to datetime
    try:
        fc_dates = [start_date + pd.DateOffset(months=int(month)) for month in fc_time]
    except ValueError as e:
        fc_dates = pd.date_range(start=start_date, periods=len(fc_time), freq='M')

    # Create an xarray Dataset
    ds_fc = xr.Dataset(
        {
            "transport": ("time", fc_trans)
        },
        coords={
            "time": fc_dates
        }
    )
    
    return ds_fc


def load_fc_full(input_dir, fname):

    # Load the data
    mat_file = os.path.join(input_dir, fname)
    mat_data = loadmat(mat_file)

    # Convert the data to a pandas DataFrame
    years = mat_data['Year'].flatten()
    months = mat_data['Month'].flatten()
    days = mat_data['Day'].flatten()
    transport = mat_data['Transport'].flatten()
    flag = mat_data['Flag'].flatten()
    df = pd.DataFrame({'Date': pd.to_datetime({'year': years, 'month': months, 'day': days}),
                       'Transport': transport, 'Flag': flag})
    df = df.dropna(subset=['Transport'])  # Remove rows with NaN in Transport for plotting

    # Set 'Date' column as the index
    df.set_index('Date', inplace=True)

    # Convert the DataFrame to an xarray Dataset
    ds_fc = df.to_xarray()
    return ds_fc