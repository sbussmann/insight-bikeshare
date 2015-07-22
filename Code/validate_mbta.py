"""

Validate MBTA score.  Plot location of MBTA stops in red with size proportional
to daily ride rate.  Overplot hubway station locations in blue with size
proportional to ride score.

"""

import pandas as pd
from pylab import savefig
import matplotlib.pyplot as plt


ridedata = pd.read_csv('../Data/Boston/FeaturesGroup3.csv')
mbtadata = pd.read_csv('../Data/Boston/mbtarideratelocation.csv')

plt.scatter(ridedata['lng'], ridedata['lat'], s=ridedata['originsubway']/5e2,
        c='blue', alpha=0.4, 
        label='Hubway station, sized by T proximity score') 
plt.scatter(mbtadata['longitude'], mbtadata['latitude'],
        s=mbtadata['ridesperday']/2e2, c='red', alpha=0.4, 
        label='MBTA subway stops, sized by average daily riders') 

plt.axis([-71.15,-71.04,42.3,42.42])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

plt.legend(loc='best')

savefig('../Figures/ValidateMBTAGroup3.png')
