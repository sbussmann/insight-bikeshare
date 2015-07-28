import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
import matplotlib.pyplot as plt
from pylab import savefig
import seaborn as sns


groupnum = 'Group5'

ridedata = pd.read_csv('../Data/Boston/Features' + groupnum + '.csv')

# stations with very numbers of rides are tourist attactions.  I want to model
# the ordinary stations.
verypopular = 45
ordinary = ridedata[ridedata['ridesperday'] < verypopular]

y = ordinary['ridesperday'].values

X = ordinary[[ 'destpop', 'destwork', 'destsubway']].values


# use linear regression
clf = linear_model.LinearRegression()

# compute r2 score using 5-fold cross-validation
scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='r2')

print(scores, scores.mean(), scores.std())

# plot correlations between features
#plt.clf()
#sns.set(style="ticks", color_codes=True)
#ordinaryfeat = ordinary[['originpop', 'originwork', 'destpop', 'destwork',
#'originsubway', 'destsubway', 'ridesperday']]
#ordinaryfeat = ordinaryfeat.sort('ridesperday')
#g = sns.pairplot(ordinaryfeat, hue="ridesperday", palette="Blues", size=4)
#plt.tight_layout()
#savefig('../Figures/FeatureCorrelation_' + groupnum + '.png')

# plot predicted number of rides vs. observed number of rides
plt.clf()
plt.figure(figsize=(6,6))
for isim in range(10):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
         X, y, test_size=0.2)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.plot([0,40], [0,40])
    plt.xlabel('Average number of rides per day', fontsize='xx-large')
    plt.ylabel('Predicted average number of rides per day', fontsize='xx-large')
savefig('../Figures/ridesperdayregression_' + groupnum + '.png')
import pdb; pdb.set_trace()
