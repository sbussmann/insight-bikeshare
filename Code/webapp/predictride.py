"""

Measure density within "radius" of "latitude, longitude"

Output: Quantitative measure of origin / destination density.

"""

import pandas as pd
from geopy.distance import vincenty
import numpy as np
import matplotlib.mlab as mlab
from sklearn import linear_model
from scipy.stats import percentileofscore
#from sklearn import cross_validation


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

def predict(inlong, inlat):

    population = pd.read_csv('../Data/Boston/population.csv')
    employee = pd.read_csv('../Data/Boston/employee.csv')

    # Generate a regular grid to interpolate the data.
    xmin = 42.29
    xmax = 42.43
    ymin = -71.19
    ymax = -70.98
    nx = 200
    ny = 200
    xvec = np.linspace(xmin, xmax, nx)
    yvec = np.linspace(ymin, ymax, ny)
    xi, yi = np.meshgrid(xvec, yvec)
    x = employee['latitude']
    y = employee['longitude']
    z = employee['EMP']
    workmap = mlab.griddata(x, y, z, xi, yi, interp='linear')
    x = population['latitude']
    y = population['longitude']
    z = population['SUBHD0401']
    popmap = mlab.griddata(x, y, z, xi, yi, interp='linear')
    #origdest = origin * destination

    radius = 2.0
    measure = getval(xvec, yvec, popmap, workmap, inlat, inlong, radius)
    popmeasure = measure[0]
    workmeasure = measure[1]

    ridedata = pd.read_csv('../Data/Boston/BostonFeaturesByStation.csv')
    y = ridedata['ridesperday'].values
    X = ridedata[['popdensity', 'workdensity']].values

    X_test = [popmeasure, workmeasure]
    clf = linear_model.LinearRegression()
    y_pred = clf.fit(X, y).predict(X_test)
    nrides_pred = y_pred[0]
    ranking = percentileofscore(y, nrides_pred)
    #clf.fit(X_train, y_train)
    #clf.coef_
    #clf.score(X_test, y_test)
    #scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='median_absolute_error')


    #print 'The predicted number of rides is %i' % nrides_pred
    return (nrides_pred, ranking)


