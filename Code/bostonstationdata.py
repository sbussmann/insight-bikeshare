import pandas as pd
#import numpy as np
import densitymetric
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


""" User Defined Parameters """

groupnum = 'Group4'

# scale radius by which to weight complementary zip codes
zipscale = 0.5

# scale radius by which to weight complementary hubway stations
stationscale = 1.0

# scale radius by which to weight complementary subway stops
subwayscale = 0.25

popemp = pd.read_csv('../Data/Boston/popemp.csv')
station = pd.read_csv('../Data/Boston/hubway_stations.csv')
mbta = pd.read_csv('../Data/Boston/mbtarideratelocation.csv')

stationlat = station['lat'].values
stationlong = station['lng'].values

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

station.to_csv('../Data/Boston/Station' + groupnum + '.csv')

databystation = pd.read_csv('../Data/Boston/HubwayRidesDays.csv')
station = pd.read_csv('../Data/Boston/Station' + groupnum + '.csv')
station = station.rename(columns = {'id': 'stationid'})
station = station.drop('Unnamed: 0', axis=1)
databystation = databystation.merge(station, on='stationid')
databystation = databystation.drop(['terminal', 'station', 'status', 'municipal'], axis=1)
databystation = databystation.drop('Unnamed: 0', axis=1)
databystation['ridesperday'] = databystation['nrides'] / databystation['ndays']
databystation.to_csv('../Data/Boston/Features' + groupnum + '.csv')

import pdb; pdb.set_trace()
