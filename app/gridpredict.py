"""

Compute predicted rides per day for each latitude and longitude in Greater
Boston area.

"""

import numpy as np
import densitymetric
import pandas as pd
#from sklearn import cross_validation
from sklearn import linear_model
import loadutil
from subprocess import call
import matplotlib.pyplot as plt
from pylab import savefig


def getorigin(ilat, ilong, popemp, mbta, zipscale, subwayscale):

    latbyzip = popemp['latitude'].values
    longbyzip = popemp['longitude'].values
    popbyzip = popemp['HD01'].values
    workbyzip = popemp['EMP'].values
    # fix nans
    finite = workbyzip * 0 == 0
    mwork = workbyzip[finite].mean()
    nans = workbyzip * 0 != 0
    workbyzip[nans] = mwork
    latbysubway = mbta['latitude']
    longbysubway = mbta['longitude']
    subwayrides = mbta['ridesperday'].values    
    distancevec = densitymetric.distvec(latbyzip, longbyzip, ilat, ilong)
    couplingzip = densitymetric.zipcouple(distancevec, zipscale)
    nzip = len(workbyzip)
    originpop = np.sum(popbyzip * couplingzip) / nzip
    originwork = np.sum(workbyzip * couplingzip) / nzip

    distancevecsubway = densitymetric.distvec(latbysubway, longbysubway, 
            ilat, ilong)

    # coupling efficiency between this station and all subway stops
    couplingsubway = densitymetric.zipcouple(distancevecsubway, subwayscale)

    # weighted sum of subway rides
    nsubway = len(subwayrides)
    originsubway = np.sum(subwayrides * couplingsubway) / nsubway

    return originpop, originwork, originsubway

def getdestination(ilat, ilong, station, stationscale,
        originpop, originwork, originsubway):

    # compute destination scores for population, employee, and subway
    destpop = []
    destwork = []
    destsubway = []

    stationlat = station['lat'].values
    stationlong = station['lng'].values
    distancevec = densitymetric.distvec(stationlat, stationlong, ilat, ilong)

    # origin to origin coupling
    stationcoupling = densitymetric.stationcouple(distancevec)

    # i may need to investigate maxcouple
    maxcouple = stationcoupling.max()

    norigin = len(originpop)
    destpop = np.sum(originpop * stationcoupling) / norigin
    destwork = np.sum(originwork * stationcoupling) / norigin
    destsubway = np.sum(originsubway * stationcoupling) / norigin

    return destpop, destwork, destsubway, maxcouple

def getfeature(ilat, ilong, popemp, mbta, station, zipscale, stationscale,
        subwayscale, stationpop, stationwork, stationsubway):

    """

    Get the feature vector associated with the input latitude and longitude.

    """


    originpop, originwork, originsubway = getorigin(ilat, ilong, popemp, mbta, 
            zipscale, subwayscale)
    destpop, destwork, destsubway, maxcouple = getdestination(ilat, ilong, 
            station, stationscale, stationpop, stationwork, stationsubway)

    features = [originpop, originwork, originsubway, destpop, destwork,
            destsubway]

    return features, maxcouple
    
def predictride(features, stationfeatures):

    # use linear regression
    clf = linear_model.LinearRegression()
    
    y = stationfeatures['ridesperday'].values

    X = stationfeatures[['originpop', 'originwork', 'originsubway', \
            'destpop', 'destwork', 'destsubway']].values
    
    nrides = clf.fit(X, y).predict(features)

    return nrides

def getride(ilat, ilong, popemp, mbta, station, zipscale, stationscale, 
        subwayscale, stationpop, stationwork, stationsubway, stationfeatures):
    ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
            station, zipscale, stationscale, subwayscale, stationpop, 
            stationwork, stationsubway)
    iride = predictride(ifeatures, stationfeatures)
    #cannibalmap[i, j] = icannibal

    # account for cannibalism
    iride = iride[0]  - iride[0] * icannibal

    return iride

def getpercentile(nride, stationfeatures):

    """ 
    
    Get the number of existing Hubway stations that have lower daily rides per
    day than the predicted number of daily rides at the given location.
    
    """

    indx = stationfeatures['ridesperday'] < nride
    place = len(stationfeatures[indx])
    return place

def makemap(groupnum = 'Group5'):

    # Generate a sub grid of latitudes and longitudes
    latvec, longvec = loadutil.grid()
    nlat = len(latvec)
    nlong = len(longvec)

    # get the mask
    mask = loadutil.getmask()
    #import matplotlib.pyplot as plt
    #extent = [longvec.min(), longvec.max(), latvec.min(), latvec.max()]
    #plt.imshow(mask, extent=extent, origin='lower')
    ##plt.scatter(biglong, biglat)
    #plt.show()
    #import pdb; pdb.set_trace()
    

    # load the data
    loaddata = loadutil.load()
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationpop = loaddata[6]
    stationwork = loaddata[7]
    stationsubway = loaddata[8]
    stationfeatures = loaddata[9]

    nrides = []
    latlist = []
    longlist = []
    for i in range(nlat):
        ilat = latvec[i]
        for j in range(nlong):
            ilong = longvec[j]
            #print("we're actually doing something!")
            iride = getride(ilat, ilong, popemp, mbta, station, zipscale, 
                    stationscale, subwayscale, stationpop, stationwork, 
                    stationsubway, stationfeatures)
            if iride > 10:
                print(i, j, ilat, ilong, iride)

            nrides.append(iride)
            latlist.append(ilat)
            longlist.append(ilong)

    nrides = np.array(nrides)
    nrides *= (1 - mask)
    nrides = list(nrides)

    ridedatadict = {'nrides': nrides, 'latitude': latlist, 'longitude':
            longlist}
    ridedata = pd.DataFrame(ridedatadict)
    ridedata.to_csv('../Data/Boston/nridesmap' + groupnum + '.csv')
    

def remakemap(ilat, ilong, iterstring):

    """

    Remake the map for a subgrid of 10x10 cells, since after adding a single
    station the impact on the predicted daily rides should be localalized.

    """

    # Generate a sub grid of latitudes and longitudes
    latvec, longvec = loadutil.grid()

    # Generate a sub grid of latitudes and longitudes
    sublatvec, sublongvec = loadutil.subgrid(ilat, ilong, nsub=10)

    # load the data
    loaddata = loadutil.load(iterstring=iterstring)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationpop = loaddata[6]
    stationwork = loaddata[7]
    stationsubway = loaddata[8]
    stationfeatures = loaddata[9]

    # read in ride map from previous iteration
    iternum = np.int(iterstring)
    iternum -= 1
    iterprevious = str(iternum)
    nrides = pd.read_csv('../Data/Boston/nridesmap_iteration' + \
            iterprevious + '.csv')
    #cannibalmap = np.zeros([nlat, nlong])
    #frides = open('../Data/Boston/nridesmap.csv', 'w')
    import time
    currenttime = time.time()
    ngrid = len(nrides)
    for i in range(ngrid):
        ilat = nrides['latitude'][i]
        ilong = nrides['longitude'][i]

        #if ilat > 42.34:
        #    if ilong > -71.09:
        #        print(ilat, ilong)
        #        import pdb; pdb.set_trace()
        #print(ilat, ilong)
        # skip this latitude if it isn't present in the subgrid
        if ilat < sublatvec.min():
            continue
        if ilat > sublatvec.max():
            continue
        # skip this longitude if it isn't present in the subgrid
        if ilong < sublongvec.min():
            continue
        if ilong > sublongvec.max():
            continue

        #print("we're actually doing something!")
        iride = getride(ilat, ilong, popemp, mbta, station, zipscale, 
                stationscale, subwayscale, stationpop, stationwork, 
                stationsubway, stationfeatures)
        nrides['nrides'].values[i] = iride
        #fmt = '{0:2} {1:9.5f} {2:9.5f} {3:8.1f} {4:8.1f} {5:8.1f} {6:8.1f} {7:8.1f} {8:8.1f} {9:6.1f}'
        #if iride > 10:
        #    print(fmt.format(i, ilat, ilong, ifeatures[0], ifeatures[1],
        #    ifeatures[2], ifeatures[3], ifeatures[4], ifeatures[5], iride))

    newtime = time.time()
    runtime = newtime - currenttime
    print("Took %d seconds to re-process the map." % runtime)
    nrides.to_csv('../Data/Boston/nridesmap_iteration' + iterstring + '.csv', \
            index=False)

    #frides.close()

def peakfind(iterstring):

    """ 
    
    Find the lat/long location with the highest predicted daily rides. 
    
    """

    ridedf = pd.read_csv('../Data/Boston/nridesmap_iteration' + \
            iterstring + '.csv')
    latmap = ridedf['latitude']
    longmap = ridedf['longitude']
    ridemap = ridedf['nrides']

    maxindx = np.argmax(ridemap)
    latmax = latmap[maxindx]
    longmax = longmap[maxindx]

    return latmax, longmax

def addnewstation(station, ilat, ilong, iterstring):

    """

    Add a row to station dataframe with location of new station.

    """

    newdic = {'lat': [ilat], 'lng': [ilong], 'status': ['proposed']}
    df1 = pd.DataFrame(newdic)
    station = station.append(df1)
    station.to_csv('../Data/Boston/hubway_stations_iteration' + iterstring + \
            '.csv', index=False)
    return 

def updatefeatures(stationfeatures, features, nrides, groupnum, iterstring):

    """

    Add a row to stationfeatures dataframe with features of new station.

    """

    maxstationid = stationfeatures['stationid'].values.max() + 1
    newdic = {'ridesperday': [nrides], 'originpop': [features[0]], \
            'originwork': [features[1]], 'originsubway': [features[2]], \
            'destpop': [features[3]], 'destwork': [features[4]], \
            'destsubway': [features[5]], 'stationid': [maxstationid]}
    df1 = pd.DataFrame(newdic)
    stationfeatures = stationfeatures.append(df1)
    stationfeatures.to_csv('../Data/Boston/Features' + groupnum + \
            '_iteration' + iterstring + '.csv', index=False)

    return

def plotmap(iterstring, groupnum='Group5'):

    # plot predicted ride map
    nrides = pd.read_csv('../Data/Boston/nridesmap_iteration' + \
            iterstring + '.csv')
    longmin = nrides['longitude'].min()
    longmax = nrides['longitude'].max()
    latmin = nrides['latitude'].min()
    latmax = nrides['latitude'].max()
    nlat = np.sqrt(np.float(len(nrides)))
    ridemap = nrides['nrides'].values.reshape((nlat, nlat))

    plt.clf()
    plt.imshow(ridemap, vmin=0, vmax=40, cmap="Blues",
            extent=[longmin,longmax,latmin,latmax], origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Predicted Daily Rides')

    # plot existing Hubway stations
    station = pd.read_csv('../Data/Boston/hubway_stations_iteration' + \
            iterstring + '.csv')
    stationfeatures = pd.read_csv('../Data/Boston/Features' + groupnum + \
            '_iteration' + iterstring + '.csv')
    plt.scatter(station['lng'], station['lat'], 
            s=stationfeatures['ridesperday'], alpha=0.4, 
            color='white', edgecolor='black', 
            label='Existing Hubway stations')
    stationnew = station[station['status'] == 'proposed']
    stationfeaturesnew = stationfeatures[station['status'] == 'proposed']
    plt.scatter(stationnew['lng'], stationnew['lat'], 
            s=stationfeaturesnew['ridesperday'], alpha=0.4, 
            color='red', edgecolor='black', 
            label='Proposed Hubway stations')
    plt.axis([longmin, longmax, latmin, latmax])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.legend()
    plt.tight_layout()
    plt.show()
    savefig('../Figures/predictedridemap_iteration.png')
    

def giveninput(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationpop, stationwork, 
            stationsubway, stationfeatures, iterstring, groupnum='Group5'):

    # predict the number of daily rides for this location
    nrides = getride(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationpop, stationwork, 
            stationsubway, stationfeatures)

    # compute how many existing stations would be worse than this station
    place = getpercentile(nrides, stationfeatures)

    # update the counter
    iternum = np.int(iterstring)
    iternum += 1
    iterstring = str(iternum)

    # add the new station, 
    addnewstation(station, ilat, ilong, iterstring)

    # recompute stationfeatures
    ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
            station, zipscale, stationscale, subwayscale, stationpop, 
            stationwork, stationsubway)
    updatefeatures(stationfeatures, ifeatures, nrides, groupnum, iterstring)

    # update the grid of predicted rides
    remakemap(ilat, ilong, iterstring)

    # remake the image showing the predicted daily rides
    plotmap(iterstring)

    return nrides, place, iterstring

def autoinput(iterstring):

    # load the data
    loaddata = loadutil.load(iterstring=iterstring)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationpop = loaddata[6]
    stationwork = loaddata[7]
    stationsubway = loaddata[8]
    stationfeatures = loaddata[9]

    ilat, ilong = peakfind(iterstring)

    nrides, place, iterstring = giveninput(ilat, ilong, popemp, mbta, station,
            zipscale, stationscale, subwayscale, stationpop, stationwork,
            stationsubway, stationfeatures, iterstring)

    return ilat, ilong, nrides, place, iterstring

def userinput(ilat, ilong, iterstring):

    # load the data
    loaddata = loadutil.load(iterstring=iterstring)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationpop = loaddata[6]
    stationwork = loaddata[7]
    stationsubway = loaddata[8]
    stationfeatures = loaddata[9]

    nrides, place, iterstring = giveninput(ilat, ilong, popemp, mbta, station,
            zipscale, stationscale, subwayscale, stationpop, stationwork,
            stationsubway, stationfeatures, iterstring)

    return ilat, ilong, nrides, place, iterstring

def resetiteration(groupnum='Group5'):

    """

    Remove all new stations from the database and start over.

    """

    dataloc = '../Data/Boston/'
    cmd = 'rm -f ' + dataloc + '*iteration*'
    call(cmd, shell=True)

    cmd = 'cp ' + dataloc + 'nridesmap' + groupnum + '.csv ' + dataloc + \
            'nridesmap' + groupnum + '_iteration0.csv'
    call(cmd, shell=True)

    cmd = 'cp ' + dataloc + 'Station' + groupnum + '.csv ' + dataloc + \
            'Station' + groupnum + '_iteration0.csv'
    call(cmd, shell=True)

    cmd = 'cp ' + dataloc + 'Features' + groupnum + '.csv ' + dataloc + \
            'Features' + groupnum + '_iteration0.csv'
    call(cmd, shell=True)

    cmd = 'cp ' + dataloc + 'hubway_stations.csv ' + dataloc + \
            'hubway_stations_iteration0.csv'
    call(cmd, shell=True)
