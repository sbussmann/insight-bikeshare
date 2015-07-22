"""

Validate population, employee, and mbta scores.  Plot location of Boston area
zip codes with size proportional to population size, employee size, and
location of mbta stops with size proportional to average rides per day at that
station.  Overplot hubway station locations in blue with size proportional to
population score, employee score, and mbta score.

"""

import pandas as pd
from pylab import savefig
import matplotlib.pyplot as plt


groupnum = 'Group4'

ridedata = pd.read_csv('../Data/Boston/Features' + groupnum + '.csv')

# population data
popdata = pd.read_csv('../Data/Boston/population.csv')

plt.clf()
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

savefig('../Figures/ValidatePopulation' + groupnum + '.png')

# employee data
empdata = pd.read_csv('../Data/Boston/employee.csv')

plt.clf()
plt.scatter(ridedata['lng'], ridedata['lat'], s=ridedata['originwork']/1e3,
        c='blue', alpha=0.4, 
        label='Hubway station, sized by employee proximity score') 
plt.scatter(empdata['longitude'], empdata['latitude'],
        s=empdata['EMP']/2e2, c='red', alpha=0.4, 
        label='Zip code centers, sized by 2012 census employee size') 

plt.axis([-71.15,-71.04,42.3,42.42])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

plt.legend(loc='best')

savefig('../Figures/ValidateEmployee' + groupnum + '.png')

# mbta data
mbtadata = pd.read_csv('../Data/Boston/mbtarideratelocation.csv')

plt.clf()
plt.scatter(ridedata['lng'], ridedata['lat'], s=ridedata['originsubway']/5e2,
        c='blue', alpha=0.4, 
        label='Hubway station, sized by MBTA proximity score') 
plt.scatter(mbtadata['longitude'], mbtadata['latitude'],
        s=mbtadata['ridesperday']/2e2, c='red', alpha=0.4, 
        label='MBTA subway stops, sized by average daily riders') 

plt.axis([-71.15,-71.04,42.3,42.42])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

plt.legend(loc='best')

savefig('../Figures/ValidateMBTA' + groupnum + '.png')


