import pandas as pd
#import numpy as np
import densitymetric
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


""" User Defined Parameters """

# scale radius by which to weight complementary zip codes
zipscale = 0.5

# scale radius by which to weight complementary hubway stations
stationscale = 1.0

# scale radius by which to weight complementary subway stops
subwayscale = 0.25

popemp = pd.read_csv('../Data/Boston/popemp.csv')
station = pd.read_csv('../Data/Boston/original/hubway_stations.csv')
mbta = pd.read_csv('../Data/Boston/mbtarideratelocation.csv')

stationlat = station['lat'].values
stationlong = station['lng'].values

scores = densitymetric.getscores(popemp, mbta, station, zipscale, 
        stationscale, subwayscale)
    #latbyzip, longbyzip, latbysubway,
    #    longbysubway, popbyzip, workbyzip, subwayrides, stationlat, 
    #    stationlong, zipscale, stationscale, subwayscale)

#plt.hist(scores[0], alpha=0.5)
#plt.hist(scores[1], alpha=0.5)
plt.scatter(scores[0], scores[1])
plt.xlabel('Origin Population Score')
plt.ylabel('Origin Employee Score')
plt.show()
plt.clf()
plt.scatter(scores[3], scores[4])
plt.xlabel('Destination Population Score')
plt.ylabel('Destination Employee Score')
plt.show()

station['originpop'] = scores[0]
station['originwork'] = scores[1]
station['originsubway'] = scores[2]
station['destpop'] = scores[3]
station['destwork'] = scores[4]
station['destsubway'] = scores[5]

station.to_csv('../Data/Boston/original/Station.csv')

databystation = pd.read_csv('../Data/Boston/HubwayRidesDays.csv')
#station = pd.read_csv('../Data/Boston/original/Station.csv')
station = station.rename(columns = {'id': 'stationid'})
#station = station.drop('Unnamed: 0', axis=1)
databystation = databystation.merge(station, on='stationid')
databystation = databystation.drop(['terminal', 'station', 'status', 'municipal'], axis=1)
databystation = databystation.drop('Unnamed: 0', axis=1)
databystation['ridesperday'] = databystation['nrides'] / databystation['ndays']
databystation.to_csv('../Data/Boston/original/Features.csv')

import pdb; pdb.set_trace()
