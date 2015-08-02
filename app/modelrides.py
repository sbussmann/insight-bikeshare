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
#from sklearn.ensemble import RandomForestRegressor



ridedata = pd.read_csv('../Data/Boston/Features.csv')

# stations with very numbers of rides are tourist attactions.  I want to model
# the ordinary stations.
verypopular = 45
ordinary = ridedata[ridedata['ridesperday'] < verypopular]

y = ordinary['ridesperday'].values

X = ordinary[['originpop', 'originwork', 'originsubway',
'destpop', 'destwork', 'destsubway']].values

#X = np.exp(X/100)
X_scaled = preprocessing.scale(X)

# use linear regression
clf = linear_model.LinearRegression()

# compute r2 score using 5-fold cross-validation
scaledscores = cross_validation.cross_val_score(clf, X_scaled, y, cv=5,
        scoring='mean_absolute_error')
scores = cross_validation.cross_val_score(clf, X, y, cv=5,
        scoring='median_absolute_error')

print("Scores for scaled linear regression: \n", 
        scaledscores, scaledscores.mean(), scaledscores.std())
print("Scores for linear regression: \n", 
        scores, scores.mean(), scores.std())

clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
scores = cross_validation.cross_val_score(clf, X_scaled, y, cv=5, scoring='r2')
print("Scores for Ridge regression: \n", 
        scores, scores.mean(), scores.std())
clf.fit(X_scaled, y)       
print(clf.alpha_, clf.coef_)

polynomial_features = PolynomialFeatures(degree=2)
ridge_regression = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("ridge_regression", ridge_regression)])
pipeline.fit(X_scaled, y)

# Evaluate the models using crossvalidation
scores = cross_validation.cross_val_score(pipeline,
    X_scaled, y, scoring="mean_squared_error", cv=5)

print("Scores for Polynomial regression: \n", 
        scores, scores.mean(), scores.std())

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='r2')
print("Scores for Decision Tree regression: \n", 
        scores, scores.mean(), scores.std())

#clf = RandomForestRegressor(n_estimators=1000)
#clf = clf.fit(X, y)
#scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='r2')
#print("Scores for Random Forest regression: \n", 
#        scores, scores.mean(), scores.std())
#clf.fit(X, y)       
#print(clf.alpha_, clf.coef_)
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
    print(clf.coef_)
    modelcoef.append(clf.coef_)

modelcoef = np.array(modelcoef)
print(modelcoef.shape)
nfeat = len(modelcoef[0, :])
print(modelcoef.mean(axis=0))
savefig('../Figures/ridesperdayregression.png')

# plot correlations between features
#import seaborn as sns
#plt.clf()
#sns.set(style="ticks", color_codes=True)
#ordinaryfeat = ordinary[['originpop', 'originwork', 'destpop', 'destwork',
#'originsubway', 'destsubway', 'ridesperday']]
#ordinaryfeat = ordinaryfeat.sort('ridesperday')
#g = sns.pairplot(ordinaryfeat, hue="ridesperday", palette="Blues", size=4)
#plt.tight_layout()
#savefig('../Figures/FeatureCorrelation.png')
import pdb; pdb.set_trace()
