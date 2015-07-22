import numpy as np
from geopy.distance import vincenty


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

def coupling(distancevec, scaledistance):

    """

    Compute the coupling efficiency between a given location and a set of
    different locations.  Model coupling efficiency with a Gaussian.

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

def getscores(latbyzip, longbyzip, latbysubway, longbysubway, popbyzip, 
        workbyzip, subwayrides, stationlat, stationlong, zipscale, 
        stationscale, subwayscale):

    """

    Compute the metrics for population density and employee density.

    TO DO: add MBTA T stop data.

    inputs:
        latvec: vector of latitudes [numpy array]
        longvec: vector of longitudes [numpy array]
        popvec: vector of population size [numpy array]
        workvec: vector of employee size [numpy array]

    """

    # for each station, compute the population score as the weighted average of
    # all zip codes
    nstation = len(stationlat)
    nsubway = len(latbysubway)
    originpop = []
    originwork = []
    originsubway = []
    distancematrix = np.zeros([nstation, nstation])
    distancematrixsubway = np.zeros([nstation, nsubway])
    for i in range(nstation):
        inlat = stationlat[i]
        inlong = stationlong[i]
        distancevec = distvec(latbyzip, longbyzip, inlat, inlong)
        couplingzip = coupling(distancevec, zipscale)
        originpop.append(np.sum(popbyzip * couplingzip))
        originwork.append(np.sum(workbyzip * couplingzip))

        distancevecsubway = distvec(latbysubway, longbysubway, inlat, inlong)

        # coupling efficiency between this station and all subway stops
        couplingsubway = coupling(distancevecsubway, subwayscale)

        # weighted sum of subway rides
        originsubway.append(np.sum(subwayrides * couplingsubway))

        # store the distance metric for each station so we don't have to
        # recompute
        distancematrix[i, :] = distancevec
        distancematrixsubway[i, :] = distancevecsubway
        fmt = '{0:3} {1:.5f} {2:.5f} {3:.2f} {4:.2f} {5:.5f} {6:.3f} {7:.3f} {8:.3f}'
        print(fmt.format(i, inlat, inlong, originpop[i], originwork[i], 
            originsubway[i], distancematrix[i, :].min(), 
            distancematrix[i, :].max(), distancematrix[i, :].mean()))

    # test: Is there a station in stationcoupling that is ~1?  Are stations
    # that are known to be far from each other correctly assigned a low
    # coupling efficiency?

    #import pdb; pdb.set_trace()


    # population close to input station
    destpop = []

    # employees close to input station
    destwork = []

    # subway stops close to input station [shape = (142,)]
    destsubway = []

    for i in range(nstation):
        distancevec = distancematrix[i, :]

        # station to station coupling
        stationcoupling = coupling(distancevec, stationscale)

        destpop.append(np.sum(originpop * stationcoupling))
        destwork.append(np.sum(originwork * stationcoupling))
        destsubway.append(np.sum(originsubway * stationcoupling))

    scores = [originpop, originwork, originsubway, destpop, destwork,
            destsubway]
    return scores

#station['popdensity'] = popdensity
#station['empdensity'] = empdensity
