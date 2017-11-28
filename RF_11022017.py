
# coding: utf-8

# In[9]:

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from collections import Counter
from sklearn.preprocessing import Imputer
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime, timedelta


# In[2]:

# Read in Data from Thomson One Database
df = pd.read_excel("090417.xls")
acq = np.array([]) # Acquirers' tickers
tar = np.array([]) # Targets' tickers
dat = np.array([]) # Date of deal announcement
cash = np.array([]) # % of cash in the deal
stock = np.array([]) # % of stocks in the deal

#Extract acquirer and target Information for those deals where both exist
for i in range(df.shape[0]):
    if df.loc[i,'Acquiror\nTicker\nSymbol'] != '-' and df.loc[i,'Target\nTicker\nSymbol'] != '-':
        acq = np.append(acq, df.loc[i,'Acquiror\nTicker\nSymbol'])
        tar = np.append(tar, df.loc[i,'Target\nTicker\nSymbol'])
        dat = np.append(dat, df.loc[i, '  Date\nAnnounced'])
        cash = np.append(cash, df.loc[i, '% of\nCash'])
        stock = np.append(stock, df.loc[i, '% of\nStock'])
        
deal_df = pd.DataFrame({'pct_of_cash': cash.ravel(), 'pct_of_stocks': stock.ravel()})
deal_df = deal_df.replace(to_replace = '-', value = 0) # May be problematic later
deal_df = deal_df.astype(float)

####EXECUTE ONCE FOR FILE GENERATION####
#np.savetxt(r'Acquiror.txt', acq, fmt='%s', newline='\r\n')
#np.savetxt(r'Target.txt', tar, fmt='%s', newline='\r\n')


# In[3]:

# Define a helper function to convert SIC into 10-industry classification
def sic_to_division(sic_list):
    """
    input:
    sic_list: numpy array of sic codes
    
    output:
    new_sic_list: numpy array of converted industry division codes
    """
    new_sic_list = np.array([])
    def func(x):
        if x < 1000:
            return 0
        elif x < 1500:
            return 1
        elif x < 1800:
            return 2
        elif x < 4000:
            return 3
        elif x < 5000:
            return 4
        elif x < 5200:
            return 5
        elif x < 6000:
            return 6
        elif x < 6800:
            return 7
        elif x < 9000:
            return 8
        elif x < 9730:
            return 9
        else:
            return 10
    converter = np.vectorize(func)
    new_sic_list = converter(sic_list)
    return new_sic_list


# In[4]:

#################################################
### READ IN FUNDAMENTALS DATA FROM COMPUSTAT ####
#################################################
df_acq = pd.read_csv("compustat0915.csv")
df_tar = pd.read_csv("compustat0915_target.csv")
####Variables to be used as features: To be added more####
var_list = ['atq', 'cshoq', 'dlcq', 'dlttq', 'oibdpq', 'ppegtq', 'pstkrq', 'teqq', 'txditcq', 'dvpy', 'dvpspq', 'prccq']

#Imputing Missing values
var_st_ind = len(df_acq.columns)

for col_name in var_list:
    # Impute values for acquirer data
    # Now mean of columns used
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(df_acq[[col_name]])
    df_acq[[col_name]]=imp.transform(df_acq[[col_name]]).ravel()
    
    # Impute values for target data
    # Now mean of columns used
    imp2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp2 = imp2.fit(df_tar[[col_name]])
    df_tar[[col_name]]=imp2.transform(df_tar[[col_name]]).ravel()

############################################################
#### Creating new variables based on basic fundamentals ####
############################################################
feature_list = ['total_assets', 'market_value_of_assets', 'tobinq', 'ppe', 'profitability', 'leverage', 'industry', 'pct_of_cash', 'pct_of_stocks']
time_dep_feature_list = ['total_assets', 'market_value_of_assets', 'tobinq', 'ppe', 'profitability', 'leverage'] # This keeps time-dependent features
special_feature_list = ['industry'] # This one keeps only categorical or time-independent features from Compustat
deal_feature_list = ['pct_of_cash', 'pct_of_stocks']

for data in [df_acq, df_tar]: 
    data['total_assets'] = data.loc[:,'atq']
    data['market_value_of_assets'] = data.loc[:,'atq'] + (data.loc[:, 'cshoq']*data.loc[:,'prccq']) - (data.loc[:,'teqq'] + data.loc[:,'txditcq'] - data.loc[:,'pstkrq'])
    data['tobinq'] = data['market_value_of_assets']/data.loc[:,'total_assets']
    data['ppe'] = data.loc[:,'ppegtq']
    data['profitability'] = data.loc[:,'oibdpq'] / data.loc[:,'total_assets']
    data['leverage'] = (data.loc[:,'dlcq'] + data.loc[:,'dlttq'])/ data.loc[:,'total_assets']
    data['industry'] = sic_to_division(data.loc[:, 'sic'])


# In[5]:

###############################################################
#### INITIATE EMPTY DATAFRAME WIH FEATURE NAMES AS COLUMNS ####
###############################################################
col = []
num_years = 3
subj = ['acq_','tar_']

# First add time-dependent, non-categorical variables as columns
for h in range(2): # Acquirer or Target
    for feature_name in time_dep_feature_list: # Feature name
        for u in range(num_years): # Numer of Years Back
            col.append(subj[h] + feature_name + '_' + str(num_years - u)+'_Q1')
            col.append(subj[h] + feature_name + '_' + str(num_years - u)+'_Q2')
            col.append(subj[h] + feature_name + '_' + str(num_years - u)+'_Q3')
            col.append(subj[h] + feature_name + '_' + str(num_years - u)+'_Q4')

# Add categorial or time-independent variables as columns
for h in range(2):
    for special_feature in special_feature_list:
        col.append(subj[h] + special_feature)

# Add features that have a single value for one deal(e.g. % of cash in the deal)
for deal_feature in deal_feature_list:
    col.append(deal_feature)
    
X = pd.DataFrame(columns = col, index = range(acq.size))


# In[6]:

####################################################################
#### Helper Functions for Date Conversions ####
####################################################################
def date_helper(t):
    """
    input: t, a date time object with attribute year and month
    output: list of integers for year and quarter
    """
    yr = t.year
    qrt = (t.month-1)//3
    if qrt == 0:
        qrt = 4
        yr -= 1
    return [yr, qrt]


# In[11]:

#####################################################
#### GENERATE FEATURE DESIGN MATRIX FOR TRAINING ####
#####################################################
sp500 = pd.read_csv("s&p500.csv")
sp500[['Date']] = pd.to_datetime(sp500['Date'])

num_lookback_qrts = 12
X_row_ind = 0
Y = np.array([])
for i in range(X.shape[0]): #Iterate over all acquirer names
    deal_date = date_helper(dat[i])
    # filt is for acquioror and fil2 is for target
    temp_acq = df_acq[(df_acq['tic'] == acq[i]) & (df_acq['fyearq'] == deal_date[0]) & (df_acq['fqtr'] == deal_date[1])]
    temp_tar = df_acq[(df_tar['tic'] == tar[i]) & (df_tar['fyearq'] == deal_date[0]) & (df_tar['fqtr'] == deal_date[1])]
    
    ### Two if statements check wheter we have all the necessary fundamentals data for the period of interest ###
    if (temp_acq.index.size > 0 and temp_tar.index.size > 0):
        deal_date_acq_ind = temp_acq.index[0]
        deal_date_tar_ind = temp_tar.index[0]
        
        if (df_acq.loc[deal_date_acq_ind - num_lookback_qrts + 1, 'tic'] == acq[i] and
           df_tar.loc[deal_date_tar_ind - num_lookback_qrts + 1, 'tic'] == tar[i]):
            
            ins_row = []
            # First work with time-dependent features
            for feature_name in time_dep_feature_list:
                for j in range(deal_date_acq_ind - num_lookback_qrts + 1, deal_date_acq_ind + 1):
                    ins_row.append(df_acq.loc[j, feature_name])
                    
            for feature_name in time_dep_feature_list:
                for j in range(deal_date_tar_ind - num_lookback_qrts + 1, deal_date_tar_ind + 1):
                    ins_row.append(df_tar.loc[j, feature_name])
            
            # Now work with categorical or time-independent features
            for special_feature in special_feature_list:
                ins_row.append(df_acq.loc[deal_date_acq_ind, special_feature])
                
            for special_feature in special_feature_list:
                ins_row.append(df_tar.loc[deal_date_tar_ind, special_feature])
            
            
            # Add deal specific features
            for deal_feature in deal_feature_list:
                ins_row.append(deal_df.loc[i, deal_feature])
            
            # Compute quarterly return after the deal which measures the success of deal
            rtn = ((df_acq.loc[deal_date_acq_ind + 1, 'prccq'] - df_acq.loc[deal_date_acq_ind, 'prccq'])
                             /df_acq.loc[deal_date_acq_ind, 'prccq'])
            
            # Adjust the returns by subtracting corresponding period's S&P 500 return
            s = str(df_acq.loc[deal_date_acq_ind, 'datadate'])
            dt = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))

            n_s = str(df_acq.loc[deal_date_acq_ind+1, 'datadate'])
            n_dt = datetime(year=int(n_s[0:4]), month=int(n_s[4:6]), day=int(n_s[6:8]))
            
            while True:
                rows = sp500['Date'].apply(lambda x: x.year == dt.year and x.month==dt.month and x.day== dt.day)
                n_rows = sp500['Date'].apply(lambda x: x.year == n_dt.year and x.month== n_dt.month and x.day == n_dt.day)
                if sum(n_rows) == 1 and sum(rows) == 1:
                    break
                elif sum(rows) != 1:
                    dt -= timedelta(days=1)
                elif sum(n_rows) != 1:
                    n_dt -= timedelta(days=1)
                
            prev_p = sp500.loc[rows, 'Close'].item()
            next_p = sp500.loc[n_rows, 'Close'].item()
            sp_rtn = (next_p - prev_p)/prev_p
            
            # Append to Y, our variable of interst
            adj_rtn =  rtn - sp_rtn
            Y = np.append(Y, adj_rtn)
            
            X.loc[X_row_ind] = ins_row
            X_row_ind += 1


# In[12]:

# One-hot Encoding for categorical variables
X = pd.get_dummies(X, columns = ['acq_industry', 'tar_industry'], prefix = ['acq_industry', 'tar_industry'])


# In[13]:

# Truncate last rows of X that are NaN rows
delete_ind = np.where(X.isnull().any(axis=1))[0][0]
X = X.iloc[:delete_ind, :]

X.to_csv('train_mat_11022017.csv')


# In[14]:

# Normalized training data
X_norm = (X - X.mean()) / (X.max() - X.min())

# Final Training Matrix's Dimensions
print(X_norm.shape)

