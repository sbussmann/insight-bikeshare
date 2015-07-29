import numpy as np
from geopy.distance import vincenty
import gridpredict
import pandas as pd
from scipy.interpolate import interp1d


def distvec(latvec, longvec, inlat, inlong):

    """

    Compute the distance in miles from a given latitude and longitude.
    
    Input: 
        latvec: vector of latitudes [numpy 1D array]
        longvec: vector of longitudes [numpy 1D array]
        inlat: latitude of station [float]
        inlong: longitude of station [float]

    Output: 
        distancevec: distance in miles from (inlat, inlong) [numpy 1D array]

    """


    origin = (inlat, inlong)
    nlat = len(latvec)
    distancevec = np.zeros(nlat)
    for ilat in range(nlat):
        destination = (latvec[ilat], longvec[ilat])
        distancevec[ilat] = vincenty(origin, destination).miles

    return distancevec

def distmap(latvec, longvec, inlat, inlong):

    """

    Map the distance in miles from a given latitude and longitude.
    
    Input: 
        latvec: vector of latitudes [numpy array]
        longvec: vector of longitudes [numpy array]
        inlat: latitude of station [float]
        inlong: longitude of station [float]

    Output: 
        distancemap: distance in miles from (inlat, inlong) [numpy array]

    """


    origin = (inlat, inlong)
    nlat = len(latvec)
    nlong = len(longvec)
    distancemap = np.zeros([nlat, nlong])
    for ilat in range(nlat):
        for ilong in range(nlong):
            destination = (latvec[ilat], longvec[ilong])
            distancemap[ilat, ilong] = vincenty(origin, destination).miles

    return distancemap

def stationcouple(distancevec, dataloc='../Data/Boston/'):

    """

    Compute the coupling efficiency between a given location and a set of
    zip code locations.  Model coupling efficiency with a normal distribution.

    Inputs: 
        distancevec: vector of distances from given station to all other points
        in Greater Boston area [numpy array]
        scaledistance: length scale over which coupling efficiency is expected
        to decrease to 1/e of the maximum [float]

    Outputs:
        couplingfactor: coupling efficiencies for each station [list]

    """

    ridelengthdf = pd.read_csv(dataloc + 'ridelengthpdf.csv')

    # assume hubway users ride at 10 miles per hour
    avgspeed = 10.

    # ride times are given in minutes in this table
    x = ridelengthdf['ridetime'].values * avgspeed * 1./60
    x = np.append(x, 120)
    y = ridelengthdf['probability'].values
    y = np.append(y, 0)

    couplingfunction = interp1d(x, y)
    couplingfactor = couplingfunction(distancevec)
    #couplingfactor = np.exp(-0.5 * (distancevec / scaledistance) ** 2)

    return couplingfactor

def zipcouple(distancevec, scaledistance):

    """

    Compute the coupling efficiency between a given location and a set of
    zip code locations.  Model coupling efficiency with a normal distribution.

    Inputs: 
        distancevec: vector of distances from given station to all other points
        in Greater Boston area [numpy array]
        scaledistance: length scale over which coupling efficiency is expected
        to decrease by 1/e [float]

    Outputs:
        couplingfactor: coupling efficiencies for each station [list]

    """

    couplingfactor = np.exp(-0.5 * (distancevec / scaledistance) ** 2)

    return couplingfactor

def getscores(popemp, mbta, station, zipscale, stationscale, subwayscale):

    """

    Compute the scores for population, employee, and MBTA proximity.

    """

    # for each station, compute the population score as the weighted average of
    # all zip codes
    stationlat = station['lat']
    stationlong = station['lng']
    nstation = len(stationlat)
    originpop = []
    originwork = []
    originsubway = []
    for i in range(nstation):
        inlat = stationlat[i]
        inlong = stationlong[i]
        originfeatures = gridpredict.getorigin(inlat, inlong, popemp, 
                mbta, zipscale, subwayscale)
        
        originpop.append(originfeatures[0])
        originwork.append(originfeatures[1])
        originsubway.append(originfeatures[2])

    print("Finished computing origin scores.")

    # population close to input station
    destpop = []

    # employees close to input station
    destwork = []

    # subway stops close to input station [shape = (142,)]
    destsubway = []

    for i in range(nstation):
        inlat = stationlat[i]
        inlong = stationlong[i]
        destinationfeatures = gridpredict.getdestination(inlat, inlong, 
                station, stationscale, originpop, originwork, originsubway)

        destpop.append(destinationfeatures[0])
        destwork.append(destinationfeatures[1])
        destsubway.append(destinationfeatures[2])
    print("Finished computing destination scores.")

    scores = [originpop, originwork, originsubway, destpop, destwork,
            destsubway]
    return scores

#station['popdensity'] = popdensity
#station['empdensity'] = empdensity
