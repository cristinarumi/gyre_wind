import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_wind_stress_curl(ds_east, ds_central, ds_west):
    linestyles=plt.cm.tab10.colors[:3]
    time_west = ds_west['time']
    wind_west = ds_west['wind']
    time_central = ds_central['time']
    wind_central = ds_central['wind']
    time_east = ds_east['time']
    wind_east = ds_east['wind']
    
    plt.figure(figsize=(10, 4))
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    plt.plot(time_west, wind_west, color=linestyles[2], linewidth=2, label='Western Region')
    plt.plot(time_central, wind_central, color=linestyles[1], linewidth=2, label='Central Region')
    plt.plot(time_east, wind_east, color=linestyles[0], linewidth=2, label='Eastern Region')
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=10)
    plt.ylabel('Wind-stress curl (10^7 N m^-3)', fontsize=12)
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylim([-0.3, 0.3])
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().invert_yaxis()
    plt.show()

def quick_quiver(u, v, sampling_x=10, sampling_y=10, scalar=None, mag_max=None, **kwargs):
    x = u.longitude
    y = u.latitude
    slx = slice(None, None, sampling_x)
    sly = slice(None, None, sampling_y)
    sl2d = (sly, slx)
    if scalar is None:
        mag = 0.5 * (u**2 + v**2)**0.5
    else:
        mag = scalar

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, **kwargs)
    ax.set_extent([-105, 25, 20, 30], crs=ccrs.PlateCarree())  # Set the extent of the map
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='face',facecolor='white')
    ax.add_feature(land)
    ax.coastlines()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # Plot the scalar field under the land
    cbar = mag.plot(ax=ax, vmax=2, transform=ccrs.PlateCarree(), zorder=1, cbar_kwargs={'label': 'Wind Stress Curl (10$^{7}$ N m$^{-3}$)'})  
    ax.quiver(x[slx], y[sly], u[sl2d], v[sl2d], transform=ccrs.PlateCarree(), zorder=1)
    
    return fig, ax, ax.quiver(x[slx], y[sly], u[sl2d], v[sl2d], transform=ccrs.PlateCarree(), zorder=1)

def plot_wsc_time_series(ds_wsc):
    # Extracting the necessary data from ds_wsc
    time = ds_wsc['time']
    full_region_deseasonalized = ds_wsc['full_region_deseasonalized']
    full_region_1yr_filtered = ds_wsc['full_region_1yr_filtered']

    # Plotting the time series
    plt.figure(figsize=(10, 6))
    plt.plot(time, full_region_deseasonalized, color='C6', linewidth=1)
    plt.plot(time, full_region_1yr_filtered, color='k', linewidth=2.5)
    plt.xlabel('Years', fontsize=15)
    plt.ylabel('WSC (10⁷ N/m³)', fontsize=15)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.title('Time Series without seasonal cycle', fontsize=15, fontweight='bold')
    plt.gcf().autofmt_xdate() #rotate xlabels
    plt.tight_layout()
    plt.show()

def moc_full(moc):
    plt.figure(figsize=(10, 5))
    plt.plot(moc.time, moc.t_therm10, label='Thermocline Transport')
    plt.plot(moc.time, moc.t_aiw10, label='Antarctic Intermediate Water Transport')
    plt.plot(moc.time, moc.t_ud10, label='Upper Deep Transport')
    plt.plot(moc.time, moc.t_ld10, label='Lower Deep Transport')
    plt.plot(moc.time, moc.t_bw10, label='Bottom Water Transport')
    plt.plot(moc.time, moc.t_gs10, label='Gulf Stream Transport')
    plt.plot(moc.time, moc.t_ek10, label='Ekman Transport')
    plt.plot(moc.time, moc.t_umo10, label='Upper Mid-Ocean Transport')
    plt.plot(moc.time, moc.moc_mar_hc10, label='MOC at 26.5N')
    plt.xlabel('Time')
    plt.ylabel('Transport (Sv)')
    plt.title('MOC Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()


def moc_seasonal(ds_moc, seasonal_cycle_moc, seasonal_std_moc):
    fig, ax = plt.subplots(1, 2, figsize=(18, 10), sharey=True, gridspec_kw={'width_ratios': [2.5, 1]})
    months = np.arange(1, 13)

    # Plot all the time series in one subplot
    ax[0].plot(ds_moc.time, ds_moc.moc_mar_hc10, color='C3', linewidth=1, label='AMOC')
    ax[0].plot(ds_moc.time, ds_moc.t_umo10, color='C4', linewidth=1, label='UMO')
    ax[0].plot(ds_moc.time, ds_moc.t_gs10, color='C0', linewidth=1, label='Florida Straits')
    ax[0].plot(ds_moc.time, ds_moc.t_ek10, color='C2', linewidth=1, label='Ekman')

    ax[0].set_xlabel('Years', fontsize=18)
    ax[0].set_ylabel('Transport (Sv)', fontsize=18)
    ax[0].set_title('Time Series', fontsize=20, fontweight='bold')
    ax[0].grid(True)
    ax[0].tick_params(axis='both', which='major', labelsize=15)

    # Plot all the seasonal cycles in the second subplot (right side)
    ax[1].plot(months, seasonal_cycle_moc['moc_mar_hc10'], color='C3', linewidth=2, label='AMOC')
    ax[1].fill_between(months, seasonal_cycle_moc['moc_mar_hc10'] - seasonal_std_moc['moc_mar_hc10'],
                       seasonal_cycle_moc['moc_mar_hc10'] + seasonal_std_moc['moc_mar_hc10'], color='C3', alpha=0.2)
    ax[1].plot(months, seasonal_cycle_moc['t_umo10'], color='C4', linewidth=2, label='UMO')
    ax[1].fill_between(months, seasonal_cycle_moc['t_umo10'] - seasonal_std_moc['t_umo10'],
                       seasonal_cycle_moc['t_umo10'] + seasonal_std_moc['t_umo10'], color='C4', alpha=0.2)
    ax[1].plot(months, seasonal_cycle_moc['t_gs10'], color='C0', linewidth=2, label='Florida Straits')
    ax[1].fill_between(months, seasonal_cycle_moc['t_gs10'] - seasonal_std_moc['t_gs10'],
                       seasonal_cycle_moc['t_gs10'] + seasonal_std_moc['t_gs10'], color='C0', alpha=0.2)
    ax[1].plot(months, seasonal_cycle_moc['t_ek10'], color='C2', linewidth=2, label='Ekman')
    ax[1].fill_between(months, seasonal_cycle_moc['t_ek10'] - seasonal_std_moc['t_ek10'],
                       seasonal_cycle_moc['t_ek10'] + seasonal_std_moc['t_ek10'], color='C2', alpha=0.2)

    ax[1].set_xlabel('Months', fontsize=18)
    ax[1].set_title('Seasonal Cycles', fontsize=20, fontweight='bold')
    ax[1].set_xticks(np.arange(1, 13))
    ax[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=15)
    ax[1].grid(True)
    ax[1].tick_params(axis='both', which='major', labelsize=15)

    ax[0].legend(loc='upper right', fontsize=15)
    ax[1].legend(loc='upper right', fontsize=15)
    fig.autofmt_xdate()  # Rotate xlabels
    plt.tight_layout()
    plt.show()
