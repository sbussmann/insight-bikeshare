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

def getdestination(ilat, ilong, station, stationscale, zipscale, 
        stationfeatures, dataloc):

    originpop = stationfeatures['originpop'].values
    originwork = stationfeatures['originwork'].values
    originsubway = stationfeatures['originsubway'].values

    stationlat = station['lat'].values
    stationlong = station['lng'].values
    distancevec = densitymetric.distvec(stationlat, stationlong, ilat, ilong)

    # origin to origin coupling
    stationcoupling = densitymetric.stationcouple(distancevec, dataloc)

    # i may need to investigate maxcouple
    othercoupling = densitymetric.zipcouple(distancevec, zipscale)
    maxcouple = othercoupling.max()

    norigin = len(originpop)
    destpop = np.sum(originpop * stationcoupling) / norigin
    destwork = np.sum(originwork * stationcoupling) / norigin
    destsubway = np.sum(originsubway * stationcoupling) / norigin

    return destpop, destwork, destsubway, maxcouple

def getfeature(ilat, ilong, popemp, mbta, station, zipscale, stationscale,
        subwayscale, stationfeatures, dataloc):

    """

    Get the feature vector associated with the input latitude and longitude.

    """


    originpop, originwork, originsubway = getorigin(ilat, ilong, popemp, mbta, 
            zipscale, subwayscale)
    destpop, destwork, destsubway, maxcouple = getdestination(ilat, ilong, 
            station, stationscale, zipscale, stationfeatures, dataloc)

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
        subwayscale, stationfeatures, dataloc):

    ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
            station, zipscale, stationscale, subwayscale, stationfeatures, 
            dataloc)
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

def makemap(dataloc):

    # Generate a sub grid of latitudes and longitudes
    latvec, longvec = loadutil.grid()
    nlat = len(latvec)
    nlong = len(longvec)

    # get the mask
    mask = loadutil.getmask(dataloc)
    #import matplotlib.pyplot as plt
    #extent = [longvec.min(), longvec.max(), latvec.min(), latvec.max()]
    #plt.imshow(mask, extent=extent, origin='lower')
    ##plt.scatter(biglong, biglat)
    #plt.show()
    #import pdb; pdb.set_trace()
    

    # load the data
    loaddata = loadutil.load(dataloc)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationfeatures = loaddata[6]

    nrides = []
    latlist = []
    longlist = []
    for i in range(nlat):
        ilat = latvec[i]
        for j in range(nlong):
            ilong = longvec[j]
            #print("we're actually doing something!")
            iride = getride(ilat, ilong, popemp, mbta, station, zipscale, 
                    stationscale, subwayscale, stationfeatures, dataloc)
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
    ridedata.to_csv(dataloc + 'nridesmap.csv')
    

def remakemap(ilat, ilong, dataloc):

    """

    Remake the map for a subgrid of 10x10 cells, since after adding a single
    station the impact on the predicted daily rides should be localalized.

    """

    # Generate a sub grid of latitudes and longitudes
    latvec, longvec = loadutil.grid()

    # Generate a sub grid of latitudes and longitudes
    sublatvec, sublongvec = loadutil.subgrid(ilat, ilong, nsub=10)

    # load the data
    loaddata = loadutil.load(dataloc)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationfeatures = loaddata[6]

    # read in ride map from previous iteration
    nrides = pd.read_csv(dataloc + 'nridesmap.csv')
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

        iride = getride(ilat, ilong, popemp, mbta, station, zipscale, 
                stationscale, subwayscale, stationfeatures, dataloc)
        nrides['nrides'][i] = iride
        #fmt = '{0:2} {1:9.5f} {2:9.5f} {3:8.1f} {4:8.1f} {5:8.1f} {6:8.1f} {7:8.1f} {8:8.1f} {9:6.1f}'
        #if iride > 10:
        #    print(fmt.format(i, ilat, ilong, ifeatures[0], ifeatures[1],
        #    ifeatures[2], ifeatures[3], ifeatures[4], ifeatures[5], iride))

    newtime = time.time()
    runtime = newtime - currenttime
    print("Took %d seconds to re-process the map." % runtime)
    nrides.to_csv(dataloc + 'nridesmap.csv', index=False)

    #frides.close()

def peakfind(dataloc):

    """ 
    
    Find the lat/long location with the highest predicted daily rides. 
    
    """

    ridedf = pd.read_csv(dataloc + 'nridesmap.csv')
    latmap = ridedf['latitude']
    longmap = ridedf['longitude']
    ridemap = ridedf['nrides']

    maxindx = np.argmax(ridemap)
    latmax = latmap[maxindx]
    longmax = longmap[maxindx]

    return latmax, longmax

def addnewstation(station, ilat, ilong, dataloc):

    """

    Add a row to station dataframe with location of new station.

    """

    newdic = {'lat': [ilat], 'lng': [ilong], 'status': ['proposed']}
    df1 = pd.DataFrame(newdic)
    station = station.append(df1)
    station.to_csv(dataloc + '/Station.csv', index=False)
    return station

def updatefeatures(stationfeatures, features, nrides, dataloc):

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
    print(dataloc)
    stationfeatures.to_csv(dataloc + 'Features.csv', index=False)

    return stationfeatures

def plotmap(dataloc):

    # plot predicted ride map
    nrides = pd.read_csv(dataloc + 'nridesmap.csv')
    longmin = nrides['longitude'].min()
    longmax = nrides['longitude'].max()
    latmin = nrides['latitude'].min()
    latmax = nrides['latitude'].max()
    nlat = np.sqrt(np.float(len(nrides)))
    ridemap = nrides['nrides'].values.reshape((nlat, nlat))

    plt.clf()
    plt.imshow(ridemap, vmin=0, cmap="Blues",
            extent=[longmin,longmax,latmin,latmax], origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Predicted Daily Rides')

    # plot existing Hubway stations
    station = pd.read_csv(dataloc + 'Station.csv')
    stationfeatures = pd.read_csv(dataloc + 'Features.csv')
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
    savefig('../Figures/predictedridemap.png')
    

def giveninput(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationfeatures, dataloc):

    # predict the number of daily rides for this location
    nrides = getride(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationfeatures, dataloc)

    # compute how many existing stations would be worse than this station
    place = getpercentile(nrides, stationfeatures)

    print "Predicted rides = %d." % nrides
    print "%d stations have lower rides." % place

    # recompute stationfeatures
    ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
            station, zipscale, stationscale, subwayscale, stationfeatures, 
            dataloc)

    print "This station would have these features: ", str(ifeatures)

    # add the new station, 
    station = addnewstation(station, ilat, ilong, dataloc)

    # add the new features to stationfeatures
    stationfeatures = updatefeatures(stationfeatures, ifeatures, nrides, dataloc)

    print("Remaking the grid of predicted rides")
    # update the grid of predicted rides
    remakemap(ilat, ilong, dataloc)

    # remake the image showing the predicted daily rides
    plotmap(dataloc)

    return nrides, place

def autoinput(dataloc):

    # load the data
    loaddata = loadutil.load(dataloc)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationfeatures = loaddata[6]

    ilat, ilong = peakfind(dataloc)

    nrides, place = giveninput(ilat, ilong, popemp, mbta, station,
            zipscale, stationscale, subwayscale, stationfeatures, dataloc)

    return ilat, ilong, nrides, place

def userinput(ilat, ilong, dataloc):

    # load the data
    loaddata = loadutil.load(dataloc)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationfeatures = loaddata[6]

    nrides, place = giveninput(ilat, ilong, popemp, mbta, station,
            zipscale, stationscale, subwayscale, stationfeatures, dataloc)

    return ilat, ilong, nrides, place

def resetiteration(basedir, growdir):

    """

    Remove all new stations from the "growing" database and start over.

    """

    #cmd = 'rm -f ' + dataloc + '*iteration*'
    #call(cmd, shell=True)
    filestocopy = ['nridesmap.csv', 'Station.csv', 'Features.csv', \
            'popemp.csv', 'mbtarideratelocation.csv', \
            'maskmap.csv', 'ridelengthpdf.csv']

    for ifile in filestocopy:
        cmd = 'cp -f ' + basedir + ifile + ' ' + growdir + ifile
        call(cmd, shell=True)
