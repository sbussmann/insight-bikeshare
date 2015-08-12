
# coding: utf-8

# In[1]:


# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig


trips = pd.read_csv('hubway_2011_07_through_2013_11/hubway_trips.csv')

plt.clf()
duration = trips[(trips['duration'] > 60) & (trips['duration'] < 3600)]['duration'].values
plt.hist((duration/60.), bins=60)
plt.xlabel('Ride Length (Minutes)')
plt.ylabel('Number of Rides')
savefig('ridelengths.png')


# In[79]:

temp = pd.DatetimeIndex(trips['start_date'])
trips['start_day'] = temp.date
trips['start_time'] = temp.time


# In[80]:

trips['riderateBoston'] = np.ones(len(temp))
grouptrips = trips.groupby('start_day', as_index=False)
sumtrip = grouptrips.sum()
sumtrip.to_csv('../Data/Boston/sumtrip.csv')


# NYC data
nyc = pd.read_csv('NYC/alldata.csv')
nyc = nyc.rename(columns = 
        {'Trips over the past 24-hours (midnight to 11:59pm)': 'riderateNYC'})
temp = pd.DatetimeIndex(nyc['Date'])
nyc['start_day'] = temp.date

#sumtrip.join(nyc, on='start_day', how='outer')
sumtrip = sumtrip.merge(nyc, on='start_day', how='outer')

#riderate = sumtrip[['riderateBoston', 'riderateNYC_y']]
riderate = sumtrip[['start_day','riderateBoston', 'riderateNYC']]
riderate.plot(x='start_day', figsize=(12, 6))
#tripsum = grouptrips.aggregate(np.sum)


# In[82]:

#nriders = grouptrips.size()


# In[83]:



#plt.clf()
#plt.plot(nriders)
plt.xlabel('Date')
plt.ylabel('Number of users per day')
savefig('ridergrowth.png')


# In[ ]:



