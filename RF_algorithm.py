from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import export_graphviz
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime, timedelta
from operator import itemgetter

### Import Cleaned Data ###
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv", header=None)
Y = Y.values.flatten()
n = X.shape[0]

### List of Features ###
feature_list = ['total_assets', 'market_value_of_assets', 'tobinq', 'ppe', 'profitability', 'leverage', 'industry', 'pct_of_cash', 'pct_of_stocks']
time_dep_feature_list = ['total_assets', 'market_value_of_assets', 'tobinq', 'ppe', 'profitability', 'leverage'] # This keeps time-dependent features
special_feature_list = ['industry'] # This one keeps only categorical or time-independent features from Compustat
deal_feature_list = ['pct_of_cash', 'pct_of_stocks']
################################################
#### Default Run of Random Forest Algorithm ####
################################################
regr = RandomForestRegressor(random_state=0)
regr.fit(X, Y)   


# In[17]:

# This is our score vector for each feature
score_vec = regr.feature_importances_

# Get the name of n features with largest scores
def get_max_score_features(score_vec, column_names, n):
    """
    output: a dictionary with key being feature names and value being corresponding feature scores
    """
    max_ind = np.argsort(-score_vec)[0:n]
    out = OrderedDict()
    for ind in max_ind:
        out[column_names[ind]] = score_vec[ind]
    return out

score_dict = get_max_score_features(score_vec, X.columns, len(score_vec))
score_dict


# In[18]:

#####################################
#### Output Analysis Begins Here ####
#####################################

### Graph the Distribution of Scores ###
plt.hist(score_vec, bins = 'auto')
plt.xlabel('Score')
plt.title('Distribution of Feature Scores')
plt.show()
# Graph is highly skewed to the right

### Calculate Mean Score of Each Year Period Prior to Deal ###
# Recall that 3 refers to the year period 3 years prior to the deal,
# 2 refers to the year period 2 years prior to the deal, and 1 to just
# the year preceding the deal.

def period_mean_score(score_dict, num_yrs):
    out = OrderedDict()
    for i in range(num_yrs):
        search_string = '_' + str(i+1) + '_'
        out[i + 1] = np.mean([value for key, value in score_dict.items() if search_string in key.lower()])
    return out

### Calculate Mean Score of Each Feature ###
def feature_mean_score(score_dict, feature_list):
    feature_mean_score = {}
    for feature in feature_list:
        one_score = np.mean([value for key, value in score_dict.items() if feature in key.lower()])
        feature_mean_score[feature] = one_score
    out = OrderedDict(sorted(feature_mean_score.items(), key=itemgetter(1), reverse = True))
    return out

# Observe that market value of assets has a notably high score
feature_mean_score(score_dict, feature_list)


# In[19]:
"""
### Parameter-tuning for Random Forest ###
### Chossing the Best Model Parameters ###

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training. What it does is, it reuses the solutions
# from previous fits and just adds more trees to existing forest instead
# of building a whole new forest at each call.

seed = 123
ensemble_regrs = [
    ("RandomForestRegressor, max_features='sqrt'",
        RandomForestRegressor(warm_start=True, max_features="sqrt", 
                               oob_score=True,
                               random_state=seed)),
    ("RandomForestRegressor, max_features='log2'",
        RandomForestRegressor(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=seed)),
    ("RandomForestRegressor, max_features=None",
        RandomForestRegressor(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=seed))
]

# Map a regressor name to a list of (<number of trees>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_regrs)

# Range of `n_estimators` values to explore.
min_num_trees = 5
max_num_trees = 70

for label, regr in ensemble_regrs:
    for i in range(min_num_trees, max_num_trees + 1):
        regr.set_params(n_estimators=i)
        regr.fit(X, Y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - regr.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "Number of Trees" plot.
for label, regr_err in error_rate.items():gf
    xs, ys = zip(*regr_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_num_trees, max_num_trees)
plt.xlabel("Number of Trees")
plt.ylabel("OOB Error Rate")
plt.legend(loc="upper right")
plt.show()

#log2 and sqrt number of features give the best result in OOB and converge at about 50 trees.

"""
# In[20]:
seed = 123
regr = RandomForestRegressor(n_estimators = 50, max_features='sqrt',
                               random_state=seed)
regr.fit(X,Y)
score_vec = regr.feature_importances_
score_dict = get_max_score_features(score_vec, X.columns, len(score_vec))
feature_mean_score(score_dict, feature_list)

# Using square root of number of features, we confirm the significance of market value of assets
# Adding the score of percent of cash and percent of stocks, the medium of deal also seems like
# an important indicator


# In[21]:

regr = RandomForestRegressor(n_estimators = 50, max_features='log2',
                               random_state=seed)
regr.fit(X,Y)
score_vec = regr.feature_importances_
score_dict = get_max_score_features(score_vec, X.columns, len(score_vec))
feature_mean_score(score_dict, feature_list)

for tree_in_forest in regr.estimators_:
    export_graphviz(tree_in_forest, out_file=None, feature_names=X.columns, filled=True, rounded=True)

#os.system('dot -Tpng tree.dot -o tree.png')

