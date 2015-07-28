
# coding: utf-8

# In[1]:


# In[2]:

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig


trips = pd.read_csv('../Data/Boston/sumtrip.csv')


# NYC data
nyc = pd.read_csv('../Data/NYC/alldata.csv')
nyc = nyc.rename(columns = 
        {'Trips over the past 24-hours (midnight to 11:59pm)': 'riderateNYC'})
temp = pd.DatetimeIndex(nyc['Date'])
nyc['start_day'] = temp.date

trips = trips.merge(nyc, on='start_day', how='outer')

riderate = trips[['start_day','riderateBoston', 'riderateNYC']]
riderate.plot(x='start_day', figsize=(12, 6))

plt.xlabel('Date')
plt.ylabel('Number of users per day')
savefig('ridergrowth.png')


# In[ ]:



