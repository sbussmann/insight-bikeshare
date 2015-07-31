import pandas as pd
import numpy as np
import glob


def load(dataloc):

    """ Helper function to load data. """

    # scale radius by which to weight complementary zip codes
    zipscale = 0.5

    # scale radius by which to weight complementary hubway stations
    stationscale = 1.0

    # scale radius by which to weight complementary subway stops
    subwayscale = 0.25

    stationfeatures = pd.read_csv(dataloc + 'Features.csv')
    station = pd.read_csv(dataloc + 'Station.csv')

    #popular = 45
    #stationfeatures = stationfeatures[stationfeatures['ridesperday'] < popular]

    popemp = pd.read_csv(dataloc + 'popemp.csv')
    mbta = pd.read_csv(dataloc + 'mbtarideratelocation.csv')

    return popemp, mbta, station, zipscale, stationscale, \
            subwayscale, stationfeatures

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
    print(value, nsub, dvec)
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


def makemask(dataloc):

    """

    Make a mask for the Charles River.

    """

    latvec, longvec = grid()
    nlat = len(latvec)
    nlong = len(longvec)

    dlat = latvec[1] - latvec[0]
    #dlong = longvec[1] - longvec[0]
    mask = np.zeros([nlat, nlong])

    masklist = glob.glob(dataloc + 'charlesriver_*csv')
    for maskfile in masklist:
        tmpmask = np.zeros([nlat, nlong])
        maskdata = pd.read_csv(maskfile)
        biglat = maskdata['latitude']
        biglong = maskdata['longitude']

        # mask the perimeter of the Charles River
        for i in range(nlat):
            for j in range(nlong):
                ilat = latvec[i]
                ilong = longvec[j]
                offlat = np.abs(biglat - ilat)
                offlong = np.abs(biglong - ilong)
                offdist = np.sqrt(offlat ** 2 + offlong ** 2)
                #if offdist.min() < 0.01:
                #    print(offdist.min())
                if offdist.min() < dlat:
                    tmpmask[i, j] = 1

        # fill in the mask
        for i in range(nlat):
            imask = tmpmask[i, :]
            masked = np.where(imask == 1)
            if masked[0].size > 1:
                mask[i, masked[0][0]: masked[0][-1]] = 1

        print("Done with " + maskfile)

    #import matplotlib.pyplot as plt
    #extent = [longvec.min(), longvec.max(), latvec.min(), latvec.max()]
    #plt.imshow(mask, extent=extent, origin='lower')
    #plt.scatter(biglong, biglat)
    #plt.show()
    #import pdb; pdb.set_trace()

    masklist = []
    for i in range(nlat):
        for j in range(nlong):
            masklist.append(mask[i, j])

    mask_df = pd.DataFrame({"mask": masklist})
    mask_df.to_csv(dataloc + 'maskmap.csv')

    return mask

def getmask(dataloc):


    latvec, longvec = grid()
    mask_df = pd.read_csv(dataloc + 'maskmap.csv')
    #maskmap = np.zeros([nlat, nlong])
    maskmap = mask_df['mask'].values

    return maskmap

