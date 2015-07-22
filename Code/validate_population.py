"""

Validate population score.  Plot location of Boston area zip codes with size
proportional to population size.  Overplot hubway station locations in blue
with size proportional to population score.

"""

import pandas as pd
from pylab import savefig
import matplotlib.pyplot as plt


ridedata = pd.read_csv('../Data/Boston/FeaturesGroup3.csv')
popdata = pd.read_csv('../Data/Boston/population.csv')

plt.scatter(ridedata['lng'], ridedata['lat'], s=ridedata['originpop']/1e3,
        c='blue', alpha=0.4, 
        label='Hubway station, sized by population proximity score') 
plt.scatter(popdata['longitude'], popdata['latitude'],
        s=popdata['HD01']/2e2, c='red', alpha=0.4, 
        label='Zip code centers, sized by 2010 census population') 

plt.axis([-71.15,-71.04,42.3,42.42])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

plt.legend(loc='best')

savefig('../Figures/ValidatePopulationGroup3.png')
