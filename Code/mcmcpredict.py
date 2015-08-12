"""

Compute optimal location for N new Hubway stations in the Greater Boston area.
Use MCMC to sample over plausible latitude and longitude values, optimizing
number of rides

"""

import numpy as np
import densitymetric
import pandas as pd
#from sklearn import cross_validation
from sklearn import linear_model
import emcee
import time


def lnprior(pzero):

    """

    Function that computes the ln prior probabilities of the model parameters.

    """

    # ensure all parameters are finite
    if (pzero * 0 != 0).any():
        priorln = -np.inf
        #pass

    # Uniform priors
    priorln = 0
    npzero = len(pzero)
    plat = pzero[0:npzero/2]
    plong = pzero[npzero/2:]
    if (plat < latmin).any():
        priorln = -np.inf
    if (plat > latmax).any():
        priorln = -np.inf
    if (plong < longmin).any():
        priorln = -np.inf
    if (plong > longmax).any():
        priorln = -np.inf

    return priorln

def lnlike(pzero):

    """
    
    Function that computes the Ln likelihood of the data.  We are trying to
    maximize the total number of rides, so that means the ln-likelihood is
    defined by the total number of rides.
    
    """
    npzero = len(pzero)
    latvec = pzero[0:npzero/2]
    longvec = pzero[npzero/2:]
    nlocation = len(latvec)
    totalrides = 0
    for i in range(nlocation):
        ilat = latvec[i]
        ilong = longvec[i]
        ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
                station, zipscale, stationscale, subwayscale, stationpop, 
                stationwork, stationsubway, latvec, longvec)
        iride = predictride(ifeatures, stationfeatures)
        #cannibalmap[i, j] = icannibal

        iride = iride[0]  - iride[0] * icannibal
        totalrides += iride
    #print(totalrides)

    return totalrides

def lnprob(pzero):

    """

    Computes ln probabilities via ln prior + ln likelihood

    """

    lp = lnprior(pzero)

    if not np.isfinite(lp):
        probln = -np.inf
        return probln

    ll = lnlike(pzero)

    normalization = 1.0#2 * real.size
    probln = lp * normalization + ll
    #print(probln, lp*normalization, ll)
    
    return probln


def getfeature(ilat, ilong, popemp, mbta, station, zipscale, stationscale,
        subwayscale, stationpop, stationwork, stationsubway, latvec, longvec):

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

# Generate a regular grid of latitudes and longitudes
latmin = 42.29
latmax = 42.43
longmin = -71.19
longmax = -70.98
nlat = 100
nlong = 100
latvec = np.linspace(latmin, latmax, nlat)
longvec = np.linspace(longmin, longmax, nlong)
latmesh, longmesh = np.meshgrid(latvec, longvec)

groupnum = 'Group4'

# scale radius by which to weight complementary zip codes
zipscale = 0.5

# scale radius by which to weight complementary hubway stations
stationscale = 1.0

# scale radius by which to weight complementary subway stops
subwayscale = 0.25

stationfeatures = pd.read_csv('../Data/Boston/Features' + groupnum + '.csv')

#popular = 45
#stationfeatures = stationfeatures[stationfeatures['ridesperday'] < popular]

stationpop = stationfeatures['originpop'].values
stationwork = stationfeatures['originwork'].values
stationsubway = stationfeatures['originsubway'].values

popemp = pd.read_csv('../Data/Boston/popemp.csv')
station = pd.read_csv('../Data/Boston/hubway_stations.csv')
mbta = pd.read_csv('../Data/Boston/mbtarideratelocation.csv')

Nthreads = 1
pool = ''
nwalkers = 22
Nnew = 5
nparams = Nnew * 2

latrand = np.random.random(Nnew) * (latmax - latmin) + latmin
longrand = np.random.random(Nnew) * (longmax - longmin) + longmin

#pzero = np.append(latrand, longrand)
pzero = np.zeros((nwalkers, nparams))
for j in range(nparams/2):
    pzero[:, j] = np.random.uniform(latmin, latmax, nwalkers)
for j in range(nparams/2, nparams):
    pzero[:, j] = np.random.uniform(longmin, longmax, nwalkers)

#startindx = nlnprob
#for j in range(nparams):
#    namej = posteriordat.colnames[j + startindx]
#    pzero[:, j] = posteriordat[namej][-nwalkers:]

sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, threads=Nthreads)

currenttime = time.time()
for pos, prob, state in sampler.sample(pzero, iterations=10000):

    print("Mean acceptance fraction: {:f}".
            format(np.mean(sampler.acceptance_fraction)), 
            "\nMean lnprob and Max lnprob values: {:f} {:f}".
            format(np.mean(prob), np.max(prob)),
            "\nTime to run previous set of walkers (seconds): {:f}".
            format(time.time() - currenttime))
    currenttime = time.time()
    #ff.write(str(prob))
    #superpos = np.zeros(1 + nparams)

    #for wi in range(nwalkers):
        #superpos[0] = prob[wi]
        #superpos[1:nparams + 1] = pos[wi]
        #posteriordat.add_row(superpos)
    #posteriordat.write('posteriorpdf.fits', overwrite=True)




#nrides = np.zeros([nlat, nlong])
##cannibalmap = np.zeros([nlat, nlong])
#for i in range(nlat):
#    for j in range(nlong):
#        ilat = latvec[i]
#        ilong = longvec[j]
#        ifeatures, icannibal = getfeature(ilat, ilong, popemp, mbta, 
#                station, zipscale, stationscale, subwayscale, stationpop, 
#                stationwork, stationsubway)
#        iride = predictride(ifeatures, stationfeatures)
#        #cannibalmap[i, j] = icannibal
#
#        iride = iride[0]  - iride[0] * icannibal
#        nrides[i, j] = iride
#        fmt = '{0:2} {1:2} {2:9.5f} {3:9.5f} {4:8.1f} {5:8.1f} {6:8.1f} {7:8.1f} {8:8.1f} {9:8.1f} {10:6.1f}'
#        if iride > 10:
#            print(fmt.format(i, j, ilat, ilong, ifeatures[0], ifeatures[1],
#            ifeatures[2], ifeatures[3], ifeatures[4], ifeatures[5], iride))
