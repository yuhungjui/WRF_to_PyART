'''
Convert reflectivity from WRF output to Pyart grid object.
'''

import sys
import numpy as np
import math
import pyart as pyart
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors

import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from wrf import (getvar, vinterp, interplevel, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair)

# Get WRF output data:

filename_list = ['./wrfout_example.nc']
file_num = 0

wrf_file = Dataset(filename_list[file_num])

# Set output gird number:
output_grid_obj_num = '%03d' % file_num

# Calculate distance from lat, lon:

def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

# Get variable and interpolate:

def get_var_interpolate(wrf_file):

    wrf_var = getvar(wrf_file, 'REFL_10CM')

# print(dbz)

# Interpolate reflectivity to pressure levels:

# interp_levels = [850]
# interp_field = vinterp(wrf_file,
#                        field=wrf_dbz,
#                        vert_coord='p',
#                        interp_levels=interp_levels,
#                        extrapolate=True,
#                        field_type='none',
#                        log_p=True)

# Interpolate reflectivity to hieghts:

interp_levels = np.linspace(0,15,31)
# print(interp_levels)
interp_field = vinterp(wrf_file,
                       field=wrf_var,
                       vert_coord='ght_msl',
                       interp_levels=interp_levels,
                       extrapolate=True,
                       field_type='none'
                      )
# print(interp_field)
# print(type(interp_field))
# print(interp_field.sizes)
# print(latlon_coords(interp_field))


# In[18]:


# Reshaping data:

# interp_field = interp_field.squeeze('interp_level')
# print(interp_field.shape)

# interp_field_3d = interp_field.squeeze('interp_level')


# In[19]:


# Get model output time:
wrf_valid_datetime = interp_field['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')


# In[20]:


# # Test plot:

# # print(interp_field.sizes)
# # interp_field.plot()

# # Get the lat/lon coordinates
# lats, lons = latlon_coords(interp_field)
# # print(lats,lons)

# # Get the cartopy projection object
# dat_crs_proj = get_cartopy(interp_field)

# # fig_crs = crs.LambertConformal(central_longitude=120.9, central_latitude=23.5)
# fig_crs = dat_crs_proj
# dat_crs = crs.PlateCarree()

# # Create the figure
# fig = plt.figure(figsize=(12,12))
# ax1 = plt.axes(projection=fig_crs)

# # ax1.set_extent([118,124,21,26.5])
# ax1.coastlines()

# # plt.rcParams['font.size'] = '18'

# # Create the color table found on NWS pages.

# # dbz_levels = np.arange(5., 75., 5.)
# # dbz_rgb = np.array([[4,233,231],
# #                     [1,159,244], [3,0,244],
# #                     [2,253,2], [1,197,1],
# #                     [0,142,0], [253,248,2],
# #                     [229,188,0], [253,149,0],
# #                     [253,0,0], [212,0,0],
# #                     [188,0,0],[248,0,253],
# #                     [152,84,198]], np.float32) / 255.0
# # dbz_map, dbz_norm = from_levels_and_colors(dbz_levels, dbz_rgb, extend="max")

# clevs = [-5,0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,85]
# ccols = ['#ffffff','#98ffff','#009aff','#1919ff','#19ff19','#19cd19','#19A019','#fefe08','#ffcb00','#ff9c00','#fe0005','#c90200','#9d0000','#9a009d','#cf00d7','#ff00f7','#fdcafe']
# dbz_map, dbz_norm = from_levels_and_colors(clevs, ccols)

# # Make the plot:

# # dbz_contours = ax.contourf(interp_field_2d.coords['XLONG'],
# #                            interp_field_2d.coords['XLAT'],
# #                            to_np(interp_field_2d),
# #                            levels=dbz_levels,
# #                            cmap=dbz_map,
# #                            norm=dbz_norm,
# #                            extend="max")

# # f1_dbz_pcolors = ax1.pcolormesh(interp_field_2d.coords['XLONG'],
# #                                 interp_field_2d.coords['XLAT'],
# #                                 to_np(interp_field_2d),
# #                                 cmap=dbz_map,
# #                                 norm=dbz_norm,
# #                                 transform=dat_crs_proj)

# plot_level = 0

# f1_dbz_pcolors = ax1.pcolormesh(interp_field.coords['XLONG'],
#                                 interp_field.coords['XLAT'],
#                                 to_np(interp_field[plot_level,:,:]),
#                                 cmap=dbz_map,
#                                 norm=dbz_norm,
#                                 transform=dat_crs)

# # Add Tiawan counties:
# shp_3 = cfeature.ShapelyFeature(shpreader.Reader('./TWN_shp/TWN_CITY').geometries(), crs.PlateCarree())
# ax1.add_feature(shp_3, facecolor='none', edgecolor='black', linewidth=1.2)

# # Add the color bar
# # cb_dbz = fig.colorbar(f1_dbz_pcolors, ax=ax1, fraction=0.04)
# cb_dbz = fig.colorbar(f1_dbz_pcolors, ticks=clevs[1:17], fraction=0.04)
# cb_dbz.ax.tick_params(labelsize=14)

# # Set axes:
# fig_tit = ax1.set_title('Reflectivity (dBZ) ' + str(interp_levels[plot_level]) + 'km ' + wrf_valid_datetime.values, fontsize=24)
# # ax1.set_xlabel('Lon', fontsize=18)
# # ax1.set_ylabel('Lat', fontsize=18)
# # ax1.labelsize = 18

# ax1_gl = ax1.gridlines(draw_labels=True)
# ax1_gl.xlabels_top = False
# ax1_gl.ylabels_right = False

# # Save figure:
# # fig_name = './wrfoutdbz_' + str(interp_levels) + '_' + interp_field['Time'].dt.strftime('%Y-%m-%d_%H:%M:%S').values
# # plt.savefig(fig_name, transparent=False, edgecolor='white', bbox_inches="tight", dpi=300)


# In[21]:


# print(type(interp_field_2d))
# print(interp_field_2d)
# print(interp_field_2d['XLONG'].values[0,:])
# print(interp_field_2d['XLAT'].values[:,0])
# print(np.array(interp_field_2d['XLONG'].values.min()))
# print(np.array(interp_field_2d['XLAT'].values.min()))
# print(pyart.config.get_metadata('origin_altitude'))
# print(interp_field_2d.values.shape)
# print(distance([23.5,120],[23.5,130]))

# print(str(interp_levels[0]))
# print(interp_field_2d['Time'].dt.strftime('%Y-%m-%d_%H:%M:%S').values)


# In[22]:


# Convert to Pyart grid object:

fields = {}
fields['reflectivity'] = {'data': interp_field.values, '_FillValue': -9999.0}
# print(fields['reflectivity'])

time = pyart.config.get_metadata('grid_time')
time['data'] = [0.0]
time['units'] = 'seconds since ' + interp_field['Time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
print(time)

# grid origin location dictionaries
origin_latitude = pyart.config.get_metadata('origin_latitude')
# origin_latitude['data'] = interp_field_2d['XLAT'].values.min()
origin_latitude['data'] = [interp_field['XLAT'].values[0,0]]

origin_longitude = pyart.config.get_metadata('origin_longitude')
# origin_longitude['data'] = interp_field_2d['XLONG'].values.min()
origin_longitude['data'] = [interp_field['XLONG'].values[0,0]]

origin_altitude = pyart.config.get_metadata('origin_altitude')
# origin_altitude['data'] = [interp_levels]
# origin_altitude['units'] = 'hPa'
origin_altitude['data'] = [0.0]
origin_altitude['units'] = 'm'

x = pyart.config.get_metadata('x')
y = pyart.config.get_metadata('y')
z = pyart.config.get_metadata('z')

tmp_x = []
for lon in interp_field['XLONG'].values[0,:]:
    tmp_x.append(distance([interp_field['XLAT'].values[0,0],interp_field['XLONG'].values[0,0]],[interp_field['XLAT'].values[0,0],lon]))

x['data'] = np.array(tmp_x) * 1000

tmp_y = []
for lat in interp_field['XLAT'].values[:,0]:
    tmp_y.append(distance([interp_field['XLAT'].values[0,0],interp_field['XLONG'].values[0,0]],[lat,interp_field['XLONG'].values[0,0]]))

y['data'] = np.array(tmp_y) * 1000

z['data'] = interp_levels * 1000
z['units'] = 'm'

# print(x)
# print(z)


# In[23]:


# metadata dictionary
metadata={}
metadata['original_container'] = 'RadialSet'
metadata['site_name'] = 'N/A'
metadata['radar_name'] = 'WRF-Output'

# create radar dictionaries
radar_latitude = pyart.config.get_metadata('radar_latitude')
radar_latitude['data'] = [origin_latitude['data']]

# print(radar_latitude['data'])
# print(type(radar_latitude['data']))

radar_longitude = pyart.config.get_metadata('radar_longitude')
radar_longitude['data'] = [origin_longitude['data']]

radar_altitude = pyart.config.get_metadata('radar_altitude')
radar_altitude['data'] = [0.0]

radar_time = pyart.config.get_metadata('radar_time')
radar_time['data'] = np.array([0])
radar_time['units'] = 'seconds since ' + interp_field['Time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

radar_name = pyart.config.get_metadata('radar_name')
radar_name['data'] = np.array(['WRF-Output'])

# projection = kwargs.pop('grid_projection', None)


# In[24]:


# Make and Write out to Pyart grid object:

output_filename = './Grid_obj/grid_' + output_grid_obj_num + '.nc'

# print(fields)
# print(fields['reflectivity'])

pyart_grid_file = pyart.core.Grid(time, fields, metadata,
                                  origin_latitude, origin_longitude, origin_altitude, x, y, z,
                                  radar_latitude=radar_latitude, radar_longitude=radar_longitude,
                                  radar_altitude=radar_altitude, radar_name=radar_name,
                                  radar_time=radar_time, projection=None
                                 )
                                #.write(filename=output_filename)

# pyart_grid_file.write(filename=output_filename)
# pyart_grid_file.write(filename=output_filename, format='NETCDF4')

pyart.io.write_grid(output_filename, pyart_grid_file)
