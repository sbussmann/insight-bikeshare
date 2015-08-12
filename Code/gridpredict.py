"""

Compute predicted rides per day for each latitude and longitude in Greater
Boston area.

"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import densitymetric
import pandas as pd
from sklearn import linear_model
import loadutil
from subprocess import call


def getorigin(ilat, ilong, popemp, mbta, zipscale, subwayscale):

    """

    Compute features for where people live, work, and ride the subway.

    """

    # locations of where people live
    latbyzip = popemp['latitude'].values
    longbyzip = popemp['longitude'].values

    # locations of where people work
    popbyzip = popemp['HD01'].values
    workbyzip = popemp['EMP'].values

    # replace nans with average value
    finite = workbyzip * 0 == 0
    mwork = workbyzip[finite].mean()
    nans = workbyzip * 0 != 0
    workbyzip[nans] = mwork

    # locations of where people ride the subway
    latbysubway = mbta['latitude']
    longbysubway = mbta['longitude']
    subwayrides = mbta['ridesperday'].values    

    # get the distance from given location to all zip codes
    distancevec = densitymetric.distvec(latbyzip, longbyzip, ilat, ilong)

    # compute the coupling factor to all zip codes
    couplingzip = densitymetric.zipcouple(distancevec, zipscale)

    # use the weighted sum as the origin score
    # normalize by number of zip codes
    nzip = len(workbyzip)
    originpop = np.sum(popbyzip * couplingzip) / nzip
    originwork = np.sum(workbyzip * couplingzip) / nzip

    # get the distance from given location to all subway stops
    distancevecsubway = densitymetric.distvec(latbysubway, longbysubway, 
            ilat, ilong)

    # coupling efficiency between this station and all subway stops
    couplingsubway = densitymetric.zipcouple(distancevecsubway, subwayscale)

    # use the weighted sum as the origin score
    # normalize by number of subway stops
    nsubway = len(subwayrides)
    originsubway = np.sum(subwayrides * couplingsubway) / nsubway

    return originpop, originwork, originsubway

def getdestination(ilat, ilong, station, stationscale, zipscale, 
        stationfeatures, dataloc):

    """

    Compute features associated with all possible destinations.


    """

    # origin features for where people live, work and ride the subway
    originpop = stationfeatures['originpop'].values
    originwork = stationfeatures['originwork'].values
    originsubway = stationfeatures['originsubway'].values

    # location of existing stations
    stationlat = station['lat'].values
    stationlong = station['lng'].values

    # compute the distance in miles from input station to all existing stations
    distancevec = densitymetric.distvec(stationlat, stationlong, ilat, ilong)

    # compute the station to station coupling factor
    stationcoupling = densitymetric.stationcouple(distancevec, dataloc)

    # determine coupling factor of closest station; this is used subsequently
    # to compute the cannibalism factor: nrides -= nrides * maxcouple
    zipcoupling = densitymetric.zipcouple(distancevec, zipscale)
    maxcouple = zipcoupling.max()

    # use the weighted sum as the destination score
    # normalize by number of stations
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


    # get features associated with the input location itself
    originpop, originwork, originsubway = getorigin(ilat, ilong, popemp, mbta, 
            zipscale, subwayscale)

    # get features associated with all possible destinations
    destpop, destwork, destsubway, maxcouple = getdestination(ilat, ilong, 
            station, stationscale, zipscale, stationfeatures, dataloc)

    # store the feature vector in a list
    features = [originpop, originwork, originsubway, destpop, destwork,
            destsubway]

    return features, maxcouple
    
def predictride(features, stationfeatures):

    """

    Predict the number of rides per day for the given features by training on
    the existing set of stationfeatures.


    """

    # use linear regression
    clf = linear_model.LinearRegression()

    # features
    X = stationfeatures[['originpop', 'originwork', 'originsubway', \
            'destpop', 'destwork', 'destsubway']].values
    
    # labels
    y = stationfeatures['ridesperday'].values
    
    # number of rides per day
    nrides = clf.fit(X, y).predict(features)

    return nrides

def getride(ilat, ilong, popemp, mbta, station, zipscale, stationscale, 
        subwayscale, stationfeatures, dataloc):

    """

    Compute the predicted rides per day for the given location.

    """

    # get the features and cannibalism factor for the given location
    ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
            station, zipscale, stationscale, subwayscale, stationfeatures, 
            dataloc)

    # predict the rides per day given the features of the new location
    iride = predictride(ifeatures, stationfeatures)

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

    """

    This function is not used in the web app.  I need to move it to its own
    file.  It is used to generate the initial map of rides per day.

    """

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

    # monitor how long this recomputing the subgrid takes
    import time
    currenttime = time.time()
    ngrid = len(nrides)
    for i in range(ngrid):
        ilat = nrides['latitude'][i]
        ilong = nrides['longitude'][i]

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

        # compute predicted rides per day given the new station
        iride = getride(ilat, ilong, popemp, mbta, station, zipscale, 
                stationscale, subwayscale, stationfeatures, dataloc)

        # store the result in the nrides dataframe
        nrides['nrides'][i] = iride

    # make sure the Charles river is still masked out
    mask = loadutil.getmask(dataloc)
    nrides['nrides'] *= (1 - mask)

    # how long did it take?  should be 5-10 seconds
    newtime = time.time()
    runtime = newtime - currenttime
    print("Took %d seconds to re-process the map." % runtime)
    nrides.to_csv(dataloc + 'nridesmap.csv', index=False)

def peakfind(dataloc):

    """ 
    
    Find the lat/long location with the highest predicted daily rides. 
    
    """

    # read the predicted rides per day
    ridedf = pd.read_csv(dataloc + 'nridesmap.csv')
    latmap = ridedf['latitude']
    longmap = ridedf['longitude']
    ridemap = ridedf['nrides']

    # get the index where the max is located
    maxindx = np.argmax(ridemap)

    # get the lat and long for that index
    latmax = latmap[maxindx]
    longmax = longmap[maxindx]

    return latmax, longmax

def addnewstation(station, ilat, ilong, dataloc):

    """

    Add a row to the station dataframe with the location of the new station.

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
    stationfeatures.to_csv(dataloc + 'Features.csv', index=False)

    return stationfeatures

def giveninput(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationfeatures, dataloc):

    """

    Given a location (ilat, ilong), input data (popemp, mbta, station), model
    parameters (zipscale, stationscale, subwayscale), and the existing hubway
    station features (stationfeatures), return the number of rides per day and
    the ranking of the proposed station.

    """

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

    # update the grid of predicted rides
    print("Remaking the grid of predicted rides")
    remakemap(ilat, ilong, dataloc)

    return nrides, place

def autoinput(dataloc):

    """

    Identify the best station location and process that location.

    """

    # load the data
    loaddata = loadutil.load(dataloc)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationfeatures = loaddata[6]

    # identify the best location for a new Hubway station
    ilat, ilong = peakfind(dataloc)

    # get the predicted rides per day and ranking for this location
    nrides, place = giveninput(ilat, ilong, popemp, mbta, station,
            zipscale, stationscale, subwayscale, stationfeatures, dataloc)

    # round the number of rides per day to 1 decimal point
    nrides = np.round(nrides, decimals=1)

    return ilat, ilong, nrides, place

def userinput(ilat, ilong, dataloc):

    """

    Process a given latitude and longitude.

    """

    # load the data
    loaddata = loadutil.load(dataloc)
    popemp = loaddata[0]
    mbta = loaddata[1]
    station = loaddata[2]
    zipscale = loaddata[3]
    stationscale = loaddata[4]
    subwayscale = loaddata[5]
    stationfeatures = loaddata[6]

    # convert latitude and longitude to floats
    ilat = np.float(ilat)
    ilong = np.float(ilong)

    # get the predicted rides per day and ranking for this location
    nrides, place = giveninput(ilat, ilong, popemp, mbta, station,
            zipscale, stationscale, subwayscale, stationfeatures, dataloc)

    # round the number of rides per day to 1 decimal point
    nrides = np.round(nrides, decimals=1)

    return ilat, ilong, nrides, place

def resetiteration(basedir, growdir):

    """

    Remove all new stations from growdir and start over.

    """

    filestocopy = ['nridesmap.csv', 'Station.csv', 'Features.csv', \
            'popemp.csv', 'mbtarideratelocation.csv', \
            'maskmap.csv', 'ridelengthpdf.csv']

    for ifile in filestocopy:
        cmd = 'cp -f ' + basedir + ifile + ' ' + growdir + ifile
        call(cmd, shell=True)
