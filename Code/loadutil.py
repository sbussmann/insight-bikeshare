import pandas as pd
import numpy as np


def load(groupnum='Group4', iterstring = '0'):

    """ Helper function to load data. """

    # scale radius by which to weight complementary zip codes
    zipscale = 0.5

    # scale radius by which to weight complementary hubway stations
    stationscale = 1.0

    # scale radius by which to weight complementary subway stops
    subwayscale = 0.25

    stationfeatures = pd.read_csv('../Data/Boston/Features' + groupnum + \
            '_iteration' + iterstring + '.csv')
    station = pd.read_csv('../Data/Boston/hubway_stations' + \
            '_iteration' + iterstring + '.csv')

    #popular = 45
    #stationfeatures = stationfeatures[stationfeatures['ridesperday'] < popular]

    stationpop = stationfeatures['originpop'].values
    stationwork = stationfeatures['originwork'].values
    stationsubway = stationfeatures['originsubway'].values

    popemp = pd.read_csv('../Data/Boston/popemp.csv')
    mbta = pd.read_csv('../Data/Boston/mbtarideratelocation.csv')

    return popemp, mbta, station, zipscale, stationscale, \
            subwayscale, stationpop, stationwork, stationsubway, \
            stationfeatures

def grid():

    """ 
    
    Helper function to generate a regular grid of latitudes and longitudes.

    """

    latmin = 42.29
    latmax = 42.43
    longmin = -71.19
    longmax = -70.98
    nlat = 100
    nlong = 100
    latvec = np.linspace(latmin, latmax, nlat)
    longvec = np.linspace(longmin, longmax, nlong)
    return latvec, longvec

def findsub(vector, value, nsub):
    dvec = vector[1] - vector[0]
    subvecmin = value - nsub / 2 * dvec
    diffvecmin = np.abs(vector - subvecmin)
    minloc = diffvecmin.argmin()
    subvec = vector[minloc: minloc + nsub + 1]

    return subvec

def subgrid(lat0, long0, nsub=10):

    """ 
    
    Helper function to generate a regular sub grid of latitudes and longitudes.
    Intended to be used for re-making a portion of the predicted ride map.

    """

    biglatvec, biglongvec = grid()

    latvec = findsub(biglatvec, lat0, nsub)
    longvec = findsub(biglongvec, long0, nsub)

    return latvec, longvec
