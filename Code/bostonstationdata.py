
# coding: utf-8

# # Aim is to get zip code for each station for each day

# In[1]:

import pandas as pd
#import numpy as np
import densitymetric
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# In[2]:

# skip first 4 cells and just read in the csv files
popemp = pd.read_csv('../Data/Boston/popemp.csv')
#employee = pd.read_csv('../Data/Boston/employee.csv')
station = pd.read_csv('../Data/Boston/hubway_stations.csv')
mbta = pd.read_csv('../Data/Boston/mbtarideratelocation.csv')

# Generate a regular grid to interpolate the data.
#xmin = 42.29
#xmax = 42.43
#ymin = -71.19
#ymax = -70.98
#nx = 200
#ny = 200
#xvec = np.linspace(xmin, xmax, nx)
#yvec = np.linspace(ymin, ymax, ny)
#xi, yi = np.meshgrid(xvec, yvec)
#x = employee['latitude']
#y = employee['longitude']
#z = employee['EMP']
#workmap = mlab.griddata(x, y, z, xi, yi, interp='linear')
#x = population['latitude']
#y = population['longitude']
#z = population['HD01']
#popmap = mlab.griddata(x, y, z, xi, yi, interp='linear')
#origdest = origin * destination


# In[3]:

stationlat = station['lat'].values
stationlong = station['lng'].values

# scale radius by which to weight complementary zip codes
zipscale = 1.0

# scale radius by which to weight complementary hubway stations
stationscale = 1.0

# scale radius by which to weight complementary subway stops
subwayscale = 0.25

# latbyzip = latitudes for each zip code
latbyzip = popemp['latitude'].values

# longbyzip = longitudes for each zip code
longbyzip = popemp['longitude'].values

# latbysubway = latitudes for each subway stop
latbysubway = mbta['latitude']

# longbysubway = longitudes for each subway stop
longbysubway = mbta['longitude']

# popbyzip = population for each zip code
popbyzip = popemp['HD01'].values

# workbyzip = number of employees for each zip code
workbyzip = popemp['EMP']

# subwayrides = average number of daily mbta rides in 2013
subwayrides = mbta['ridesperday']

scores = densitymetric.getscores(latbyzip, longbyzip, latbysubway,
        longbysubway, popbyzip, workbyzip, subwayrides, stationlat, 
        stationlong, zipscale, stationscale, subwayscale)

#plt.hist(scores[0], alpha=0.5)
#plt.hist(scores[1], alpha=0.5)
plt.scatter(scores[0], scores[1])
plt.xlabel('Population Score')
plt.ylabel('Employee Score')
plt.show()

station['originpop'] = scores[0]
station['originwork'] = scores[1]
station['originsubway'] = scores[2]
station['destpop'] = scores[3]
station['destwork'] = scores[4]
station['destsubway'] = scores[5]
import pdb; pdb.set_trace()

station.to_csv('../Data/Boston/StationGroup2.csv')

