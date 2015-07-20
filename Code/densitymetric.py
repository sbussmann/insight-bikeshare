
# coding: utf-8

"""

Measure density within "radius" of "latitude, longitude"

Output: Quantitative measure of origin / destination density.

"""

import numpy as np
from geopy.distance import vincenty


def getval(xmap, ymap, popmap, workmap, inlat, inlong, radius):

    origin = (inlat, inlong)
    nlat = len(workmap[:, 0])
    nlong = len(workmap[0, :])
    distance = np.zeros([nlat, nlong])
    for ilat in range(nlat):
        for ilong in range(nlong):
            destination = (xmap[ilat], ymap[ilong])
            distance[ilat, ilong] = vincenty(origin, destination).miles
    close = np.where(distance < 2.0)

    popmeasure = np.median(popmap[close])
    workmeasure = np.median(workmap[close])
    measure = (popmeasure, workmeasure)
    #print(measure)

    return measure
