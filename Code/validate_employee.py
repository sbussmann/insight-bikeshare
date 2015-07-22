"""

Validate employee score.  Plot location of Boston area zip codes with size
proportional to number of employees.  Overplot hubway station locations in blue
with size proportional to employee score.

"""

import pandas as pd
from pylab import savefig
import matplotlib.pyplot as plt


ridedata = pd.read_csv('../Data/Boston/FeaturesGroup3.csv')
empdata = pd.read_csv('../Data/Boston/employee.csv')

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

savefig('../Figures/ValidateEmployeeGroup3.png')
