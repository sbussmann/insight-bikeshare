import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
import matplotlib.pyplot as plt
from pylab import savefig
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor



ridedata = pd.read_csv('../Data/Boston/Features.csv')

# stations with very numbers of rides are tourist attactions.  I want to model
# the ordinary stations.
verypopular = 45
ordinary = ridedata[ridedata['ridesperday'] < verypopular]

y = ordinary['ridesperday'].values

X = ordinary[['originpop', 'originwork', 'originsubway',
'destpop', 'destwork', 'destsubway']].values
X_original = ordinary[['originpop', 'originwork', 'originsubway',
'destpop', 'destwork', 'destsubway']].values

X = preprocessing.scale(X)
tmp4 = X[:,4]
tmp5 = X[:,5]
X[:, 4] = (tmp4 + tmp5) / 2.
X[:, 5] = (tmp4 - tmp5) / 2.
X = preprocessing.scale(X)
X_scaled = X.copy()

ordinary['originpop'] = X[:, 0]
ordinary['originwork'] = X[:, 1]
ordinary['originsubway'] = X[:, 2]
ordinary['destpop'] = X[:, 3]
ordinary['destwork'] = X[:, 4]
ordinary['destsubway'] = X[:, 5]

#X = np.exp(X/100)

def niceprint(modeltype, metric, scores):
    smean = str(round(scores.mean(), 2))
    srms = str(round(scores.std(), 2))
    print(modeltype + ", " + metric + ': %s +/- %s' % (smean, srms))
    return

# use linear regression
clf = linear_model.LinearRegression()

metric = "mean_absolute_error"

# compute r2 score using 5-fold cross-validation
scaledscores = cross_validation.cross_val_score(clf, X_scaled, y, cv=5,
        scoring=metric)
scores = cross_validation.cross_val_score(clf, X_original, y, cv=5,
        scoring=metric)
niceprint('Linear Regression', metric, scores)

clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
scores = cross_validation.cross_val_score(clf, X_scaled, y, cv=5,
        scoring=metric)
niceprint('Ridge Regression', metric, scores)
#clf.fit(X_scaled, y)       
#print(clf.alpha_, clf.coef_)

polynomial_features = PolynomialFeatures(degree=3)
ridge_regression = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("ridge_regression", ridge_regression)])
#pipeline.fit(X_scaled, y)

# Evaluate the models using crossvalidation
scores = cross_validation.cross_val_score(pipeline,
    X_scaled, y, scoring=metric, cv=5)

niceprint("Polynomial Ridge regression", metric, scores)

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
scores = cross_validation.cross_val_score(clf, X, y, cv=5,
        scoring=metric)
niceprint("Decision Tree regression", metric, scores)

clf = RandomForestRegressor(n_estimators=100, max_features='sqrt')
clf = clf.fit(X, y)
#print(clf.alpha_, clf.coef_)
scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring=metric)
niceprint("Random Forest Regression:", metric, scores)

clf = RandomForestRegressor(n_estimators=10000)
poly = PolynomialFeatures(degree=4)
Xpoly = poly.fit_transform(X)
clf = clf.fit(Xpoly, y)
scores = cross_validation.cross_val_score(clf, Xpoly, y, cv=5, scoring=metric)
niceprint("Random Forest Polynomial Regression:", metric, scores)
#clf.fit(X, y)       
#print("Coefficients for Polynomial regression: ")
#model = Pipeline([('poly', PolynomialFeatures(degree=3)), 
#    ('linear', LinearRegression())])
#modelfit = model.fit(X_scaled, y)
#print(clf.coef_)

# plot predicted number of rides vs. observed number of rides
clf = linear_model.LinearRegression()
plt.clf()
plt.figure(figsize=(6,6))
modelcoef = []
for isim in range(10):
    X_scaled_train, X_scaled_test, y_train, y_test = cross_validation.train_test_split(
         X_scaled, y, test_size=0.2)
    y_pred = clf.fit(X_scaled_train, y_train).predict(X_scaled_test)
    plt.scatter(y_test, y_pred)
    plt.plot([0,45], [0,45])
    plt.xlabel('True Daily Rides', fontsize='xx-large')
    plt.ylabel('Predicted Daily Rides', fontsize='xx-large')
    plt.tight_layout()
    #print(clf.coef_)
    modelcoef.append(clf.coef_)

modelcoef = np.array(modelcoef)
print(modelcoef.shape)
nfeat = len(modelcoef[0, :])
print(modelcoef.mean(axis=0))
savefig('../Figures/ridesperdayregression.png')

# plot correlations between features
import seaborn as sns
plt.clf()
sns.set(style="ticks", color_codes=True)
ordinaryfeat = ordinary[['originsubway', 'destpop', 'destwork', 'destsubway', 'ridesperday']]
#tmp1 = ordinary['destsubway'] + ordinary['destwork']
#tmp2 = ordinary['destsubway'] - ordinary['destwork']
#ordinaryfeat['destsubpluswork'] = tmp1
#ordinaryfeat['destsubminuswork'] = tmp2
ordinaryfeat = ordinaryfeat.sort('ridesperday')
g = sns.pairplot(ordinaryfeat, hue="ridesperday", palette="Blues", size=4)
plt.tight_layout()
savefig('../Figures/FeatureCorrelation.png')
import pdb; pdb.set_trace()
