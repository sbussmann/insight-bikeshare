import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from pylab import savefig


#ax = sns.heatmap(nrides, vmin=0, vmax=50, extent=[longmin,longmax,latmin,latmax])
plt.imshow(nrides, vmin=0, vmax=100, extent=[longmin,longmax,latmin,latmax], origin='lower')
plt.clf()
#my_map = Basemap(projection='merc',
#    resolution = 'f', area_thresh = 0.001,
#    llcrnrlon=-72, llcrnrlat=42,
#    urcrnrlon=-70, urcrnrlat=44)
 
#my_map.drawcoastlines()
#my_map.drawrivers()
#my_map.fillcontinents(color='coral', lake_color='aqua', zorder=0)
#my_map.drawmapboundary()
#my_map.pcolormesh(longvec, latvec, nrides, cmap='Blues', vmin=0, vmax=25, latlon=True)
#cbar = my_map.colorbar()
#cbar.set_label('Predicted number of daily rides')
my_map.scatter(station['lng'], station['lat'], s=stationfeatures['ridesperday'], alpha=0.4, color='white', edgecolor='black', label='Hubway stations, sized by \n observed number of daily rides', latlon=True)
#plt.axis([longmin, longmax, latmin, latmax])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='best')
plt.tight_layout()
savefig('../Figures/predictedridebasemap.png')
import pdb; pdb.set_trace()

