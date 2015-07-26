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


def getfeature(ilat, ilong, popemp, mbta, station, zipscale, stationscale,
        subwayscale, stationpop, stationwork, stationsubway):

    """

    Get the feature vector associated with the input latitude and longitude.

    """

    stationlat = station['lat'].values
    stationlong = station['lng'].values
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
    couplingzip = densitymetric.coupling(distancevec, zipscale)
    originpop = np.sum(popbyzip * couplingzip)
    originwork = np.sum(workbyzip * couplingzip)

    distancevecsubway = densitymetric.distvec(latbysubway, longbysubway, 
            ilat, ilong)

    # coupling efficiency between this station and all subway stops
    couplingsubway = densitymetric.coupling(distancevecsubway, subwayscale)

    # weighted sum of subway rides
    originsubway = np.sum(subwayrides * couplingsubway)

    #fmt = '{0:3} {1:.5f} {2:.5f} {3:.2f} {4:.2f} {5:.5f} {6:.3f} {7:.3f}'
    #print(fmt.format(ilat, ilong, originpop, originwork, 
    #    originsubway, distancevec.min(), 
    #    distancevec.max(), distancevec.mean()))

    # test: Is there a station in stationcoupling that is ~1?  Are stations
    # that are known to be far from each other correctly assigned a low
    # coupling efficiency?  Tests indicate yes.

    # compute destination scores for population, employee, and subway
    destpop = []
    destwork = []
    destsubway = []

    distancevec = densitymetric.distvec(stationlat, stationlong, ilat, ilong)

    # station to station coupling
    stationcoupling = densitymetric.coupling(distancevec, stationscale)

    destpop = np.sum(stationpop * stationcoupling)
    destwork = np.sum(stationwork * stationcoupling)
    destsubway = np.sum(stationsubway * stationcoupling)

    features = [originpop, originwork, destpop, destwork,
            originsubway, destsubway]

    maxcouple = stationcoupling.max()

    return features, maxcouple
    
def predictride(features, stationfeatures):

    # use linear regression
    clf = linear_model.LinearRegression()
    
    y = stationfeatures['ridesperday'].values

    X = stationfeatures[['originpop', 'originwork', 'destpop', \
            'destwork', 'originsubway', 'destsubway']].values
    
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

def makemap(iternum):

    # Generate a regular grid of latitudes and longitudes
    latvec, longvec = loadutil.grid()
    nlat = len(latvec)
    nlong = len(longvec)

    # load the data
    loaddata = loadutil.load(iternum)
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

    nrides = np.zeros([nlat, nlong])
    #cannibalmap = np.zeros([nlat, nlong])
    #frides = open('../Data/Boston/nridesmap.csv', 'w')
    latlist = []
    longlist = []
    nridelist = []
    for i in range(nlat):
        ilat = latvec[i]
        for j in range(nlong):
            ilong = longvec[j]
            ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
                    station, zipscale, stationscale, subwayscale, stationpop, 
                    stationwork, stationsubway)
            iride = predictride(ifeatures, stationfeatures)
            #cannibalmap[i, j] = icannibal

            iride = iride[0]  - iride[0] * icannibal
            nrides[i, j] = iride
            latlist.append(ilat)
            longlist.append(ilong)
            nridelist.append(iride)

    ridedict = {'nrides': nridelist, 'latitude': latlist, 'longitude':
            longlist}
    ridedf = pd.DataFrame(ridedict)
    ridedf.to_csv('../Data/Boston/ridemap_iteration' + iternum + '.csv')
            #slat = str(ilat)
            #slong = str(ilong)
            #sride = str(iride)
            #frides.write(slat + ',' + slong + ',' + sride + '\n')
            #fmt = '{0:2} {1:2} {2:9.5f} {3:9.5f} {4:8.1f} {5:8.1f} {6:8.1f} {7:8.1f} {8:8.1f} {9:8.1f} {10:6.1f}'
            #if iride > 10:
                #print(fmt.format(i, j, ilat, ilong, ifeatures[0], ifeatures[1],
                #ifeatures[2], ifeatures[3], ifeatures[4], ifeatures[5], iride))

    #frides.close()

def peakfind(iternum):

    """ 
    
    Find the lat/long location with the highest predicted daily rides. 
    
    """

    ridedf = pd.read_csv('../Data/Boston/ridemap_iteration' + \
            iternum + '.csv')
    latmap = ridedf['latitude'].reshape(100, 100)
    longmap = ridedf['longitude'].reshape(100, 100)
    ridemap = ridedf['nrides'].reshape(100, 100)

    maxindx = np.argmax(ridemap)
    latmax = latmap[maxindx]
    longmax = longmap[maxindx]

    return latmax, longmax

def addnewstation(station, ilat, ilong, iternum):

    """

    Add a row to station dataframe with location of new station.

    """

    newdic = {'lat': [ilat], 'lng': [ilong], 'station': ['proposed']}
    df1 = pd.DataFrame(newdic)
    station = station.append(df1)
    station.to_csv('../Data/Boston/hubway_station_iteration' + iternum + \
            '.csv')
    return 

def giveninput(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationpop, stationwork, 
            stationsubway, stationfeatures, iternum):

    # predict the number of rides for this location
    nrides = getride(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationpop, stationwork, 
            stationsubway, stationfeatures)

    # compute how many existing stations would be worse than this station
    place = getpercentile(nrides, stationfeatures)

    # add the new station, 
    addnewstation(station, ilat, ilong, iternum)

    # recompute stationfeatures
    updatefeatures(stationfeatures, ilat, ilong, iternum)

    # update the grid of predicted rides
    makemap(iternum)

    return nrides, place

def autoinput(iternum):

    # load the data
    loaddata = loadutil.load(iternum)
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

    ilat, ilong = peakfind(iternum)

    nrides, place = giveninput(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationpop, stationwork, 
            stationsubway, stationfeatures, iternum)

    return nrides, place

def userinput(ilat, ilong, iternum):

    # load the data
    loaddata = loadutil.load(iternum)
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

    nrides, place = giveninput(ilat, ilong, popemp, mbta, station, zipscale, 
            stationscale, subwayscale, stationpop, stationwork, 
            stationsubway, stationfeatures, iternum)

    return nrides, place
