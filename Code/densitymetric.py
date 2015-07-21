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
    popmeasures = []
    workmeasures = []
    stationcoupling = np.zeros([nstation, nstation])
    for i in range(nstation):
        inlat = stationlat[i]
        inlong = stationlong[i]
        distancemap = distmap(latvec, longvec, inlat, inlong)
        close = distancemap < closeradius
        popmeasures.append(popmap[close].mean())
        workmeasures.append(workmap[close].mean())
        stationcoupling[i, :] = coupling(latvec, longvec, distancemap, 
                stationlat, stationlong, scaledistance)
        values = (i, popmeasures[i], workmeasures[i], 
                stationcoupling[i, :].min(), stationcoupling[i, :].max(),
                stationcoupling[i, :].mean())
        fmt = '{0} {1:.2f} {2:.2f} {3:.5f} {4:.3f} {5:.3f}'
        print(fmt.format(values))

    # test: Is there a station in stationcoupling that is ~1?  Are stations
    # that are known to be far from each other correctly assigned a low
    # coupling efficiency?

    #import pdb; pdb.set_trace()


    popscore = []
    workscore = []
    for i in range(nstation):
        popscore.append(np.average(popmeasures, 
            weights=stationcoupling[i, :]))
        workscore.append(np.average(workmeasures, 
            weights=stationcoupling[i, :]))

    import pdb; pdb.set_trace()
    scores = [popscore, workscore]
    return scores

#station['popdensity'] = popdensity
#station['empdensity'] = empdensity
