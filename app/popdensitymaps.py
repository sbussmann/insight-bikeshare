
# coding: utf-8

"""
 Goal: generate a heat map of population density for both residents and employees using US Census data for the city of Boston.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pylab import savefig
#from scipy.ndimage.interpolation import map_coordinates


# read in employee data from US Census
employee = pd.read_csv('censusdata/BP_2012_00CZ1/BP_2012_00CZ1_with_ann.csv')

# greater Boston area zip codes (from freemaptools.com)
bostonzip = pd.read_csv('bostonzipcodes.csv', \
        converters={'zip': lambda x: str(x)})

# filter out non-Boston zip codes
employee = employee.merge(bostonzip, on='zip')

# get latitude and longitude of zip codes
latlong = pd.read_csv('zipcode/zipcode.csv', \
        converters={'zip': lambda x: str(x)})

# add lat and long data to employee dataframe
employee = employee.merge(latlong, on='zip')

# convert non-numeric values to NaNs
employee['EMP'] = employee['EMP'].convert_objects(convert_numeric=True)

# read in population data from US Census and merge with zip code data and
# long/lat data
population = pd.read_csv(
        'censusdata/DEC_10_SF1_GCTPH1/DEC_10_SF1_GCTPH1.ST09_with_ann.csv', 
        converters={'zip': lambda x: x[-5:]})
population = population.merge(bostonzip, on='zip')
population = population.merge(latlong, on='zip')

# plot latitude and longitude of employee locations with size determined by
# number of employees
sizevec = employee['EMP'] / 200
#employee.plot(x='longitude', y='latitude', kind='scatter', s=sizevec)

# Generate a regular grid to interpolate the data.
xmin = -71.19
xmax = -70.98
ymin = 42.29
ymax = 42.43
nx = 200
ny = 200
xi = np.linspace(xmin, xmax, nx)
yi = np.linspace(ymin, ymax, ny)
xi, yi = np.meshgrid(xi, yi)

# Interpolate using griddata
x = employee['longitude']
y = employee['latitude']
z = employee['EMP']
zi = mlab.griddata(x, y, z, xi, yi, interp='linear')
blue = zi / zi.max()
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.spines.set_color('white')
#ax.tick_params(colors='white')
#plt.pcolormesh(xi, yi, blue, cmap='Blues', shading='gouraud')

x = population['longitude']
y = population['latitude']
z = population['SUBHD0401']
zi = mlab.griddata(x, y, z, xi, yi, interp='linear')
red = zi / zi.max()
#plt.pcolormesh(xi, yi, red, cmap='Reds', alpha=0.3, shading='gouraud')

green = red.copy()
green[:] = 0

rgb = np.zeros((200, 200, 3))
rgb[:, :, 0] = red
rgb[:, :, 1] = green
rgb[:, :, 2] = blue
extent = (xmin, xmax, ymin, ymax)
plt.imshow(rgb, origin='lower', extent=extent)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

#employee.plot('EMP', 'PAYANN', kind='scatter')
levs = 2 ** (np.arange(2, 10, 0.5)) * 100
levs = levs[6:]
#plt.contour(xi, yi, zi, colors='grey')
#plt.pcolormesh(xi, yi, zi, cmap='Blues')
#plt.scatter(x, y, c=z, s=sizevec, alpha=0.2)
#plt.colorbar()
#plt.axis([xmin, xmax, ymin, ymax])

#plot boston common, fanueil hall, harvard, and MIT
lat1 = [42 + 21/60. + 16/3600., 42.36, 42.3744,42.3598]
long1 = [-71 - 03/60. - 54/3600.,-71.056667, -71.1169, -71.0921]
plt.plot(long1, lat1, 'o')

# plot Hubway stations
stations = pd.read_csv('2014-1223_RegularSeasonStationsLatitudeLongitude.csv')
plt.plot(stations['Longitude'], stations['Latitude'], '.', color='white')
plt.axis(extent)
savefig('populationrgb.png')

#plt.show()
import pdb; pdb.set_trace()
