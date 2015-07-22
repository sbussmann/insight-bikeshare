import numpy as np
from geopy.distance import vincenty


def distmap(latvec, longvec, inlat, inlong):

    """

    Map the distance in miles from a given latitude and longitude.
    
    Input: 
        xvec: high resolution vector of latitudes [numpy array]
        yvec: high resolution vector of longitudes [numpy array]
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

def coupling(latvec, longvec, distancemap, stationlat, stationlong,
        scaledistance):

    """

    Compute the coupling efficiency between a given station and all other
    stations.  Model coupling efficiency with a Gaussian.

    Inputs: 
        stationlat: station latitudes [list]
        stationlong: station longitudes [list]
        distancemap: map of distance from given station to all other points
        in Greater Boston area [numpy array]
        scaledistance: length scale over which coupling efficiency is expected
        to decrease by 1/e [float]

    Outputs:
        eta_coupling: coupling efficiencies for each station [list]

    """

    from scipy.interpolate import interp2d

    sdfunc = interp2d(latvec, longvec, distancemap, kind='cubic')
    stationdistance = []
    for i in range(len(stationlat)):
        interp = sdfunc(stationlat[i], stationlong[i])
        stationdistance.append(interp[0])
    stationdistance = np.array(stationdistance)
    stationcoupling = np.exp(-0.5 * (stationdistance / scaledistance) ** 2)

    return stationcoupling

def getscores(latvec, longvec, popmap, workmap, stationlat, stationlong,
        closeradius, scaledistance):

    """

    Compute the metrics for population density and employee density.

    TO DO: add MBTA T stop data.

    inputs:
        latvec: vector of high resolution latitudes [numpy array]
        longvec: vector of high resolution longitudes [numpy array]
        popmap: high resolution map of population density [numpy array]
        workmap: high resolution map of employee density [numpy array]
        stationcoupling: coupling efficiency

    """

    nstation = len(stationlat)
    originpop = []
    originwork = []
    stationcoupling = np.zeros([nstation, nstation])
    for i in range(nstation):
        inlat = stationlat[i]
        inlong = stationlong[i]
        distancemap = distmap(latvec, longvec, inlat, inlong)
        close = distancemap < closeradius
        originpop.append(popmap[close].mean())
        originwork.append(workmap[close].mean())
        stationcoupling[i, :] = coupling(latvec, longvec, distancemap, 
                stationlat, stationlong, scaledistance)
        fmt = '{0:3} {1:.2f} {2:.2f} {3:.5f} {4:.3f} {5:.3f}'
        print(fmt.format(i, originpop[i], originwork[i], 
                stationcoupling[i, :].min(), stationcoupling[i, :].max(),
                stationcoupling[i, :].mean()))

    # test: Is there a station in stationcoupling that is ~1?  Are stations
    # that are known to be far from each other correctly assigned a low
    # coupling efficiency?

    #import pdb; pdb.set_trace()


    destpop = []
    destwork = []
    for i in range(nstation):
        destpop.append(np.average(originpop, 
            weights=stationcoupling[i, :]))
        destwork.append(np.average(originwork, 
            weights=stationcoupling[i, :]))

    scores = [originpop, originwork, destpop, destwork]
    return scores

#station['popdensity'] = popdensity
#station['empdensity'] = empdensity
