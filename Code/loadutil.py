import pandas as pd
import numpy as np


def load(groupnum='Group4', iternum = ''):

    """ Helper function to load data. """

    # scale radius by which to weight complementary zip codes
    zipscale = 0.5

    # scale radius by which to weight complementary hubway stations
    stationscale = 1.0

    # scale radius by which to weight complementary subway stops
    subwayscale = 0.25

    stationfeatures = pd.read_csv('../Data/Boston/Features' + groupnum + \
            '_iteration' + iternum + '.csv')
    station = pd.read_csv('../Data/Boston/hubway_stations' + \
            '_iteration' + iternum + '.csv')

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

