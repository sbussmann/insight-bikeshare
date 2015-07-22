
# coding: utf-8

# # Aim is to get zip code for each station for each day

# In[1]:

import pandas as pd
import numpy as np
import densitymetric
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# In[2]:

# skip first 4 cells and just read in the csv files
population = pd.read_csv('../Data/Boston/population.csv')
employee = pd.read_csv('../Data/Boston/employee.csv')
station = pd.read_csv('../Data/Boston/hubway_stations.csv')

# Generate a regular grid to interpolate the data.
xmin = 42.29
xmax = 42.43
ymin = -71.19
ymax = -70.98
nx = 200
ny = 200
xvec = np.linspace(xmin, xmax, nx)
yvec = np.linspace(ymin, ymax, ny)
xi, yi = np.meshgrid(xvec, yvec)
x = employee['latitude']
y = employee['longitude']
z = employee['EMP']
workmap = mlab.griddata(x, y, z, xi, yi, interp='linear')
x = population['latitude']
y = population['longitude']
z = population['SUBHD0401']
popmap = mlab.griddata(x, y, z, xi, yi, interp='linear')
#origdest = origin * destination


# In[3]:

stationlat = station['lat'].values
stationlong = station['lng'].values
closeradius = 0.25
scaledistance = 1.0
scores = densitymetric.getscores(xvec, yvec, popmap, workmap, stationlat, 
        stationlong, closeradius, scaledistance)

#plt.hist(scores[0], alpha=0.5)
#plt.hist(scores[1], alpha=0.5)
plt.scatter(scores[0], scores[1])
plt.xlabel('Population density')
plt.ylabel('Employee density')
plt.show()

station['originpop'] = scores[0]
station['originwork'] = scores[1]
station['destpop'] = scores[2]
station['destwork'] = scores[3]

station.to_csv('../Data/Boston/LinkedStationDensity.csv')

import pdb; pdb.set_trace()
