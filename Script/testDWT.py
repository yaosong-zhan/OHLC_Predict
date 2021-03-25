# %%
import pandas as pd
import numpy as np
from queryData import QueryData
from dtaidistance import dtw
import sklearn.cluster as skc
from sklearn.preprocessing import normalize
# %% Define the DTW distance when the sample has multiple curves

def distance(curve1, curve2):
    dim = 4
    dist = 0
    for idx in range(dim):
        dist = dtw.distance(curve1[idx*20:(idx+1)*20], curve2[idx:(idx+1)*20]) + dist
    
    return dist

# %% Read data

filename = '../Data/SP500.csv'
query = QueryData()
data = query.readlocal(filename, 'csv')
# %% Process data

varlist = {'Close':[], 'Open':[], 'High':[], 'Low':[]}
for varname in varlist.keys():
    varlist[varname] = np.array(data[varname])
    varlist[varname] = varlist[varname][:16000].reshape((-1,20))

# %% Normalize

for varname in varlist.keys():
    varlist[varname] = normalize(varlist[varname])
# %% Construct multiple curves

mulCurves = np.empty(shape = (varlist['Close'].shape[0],len(varlist)*varlist['Close'].shape[1]))
numdays = varlist['Close'].shape[1]
mulCurves[:,0:numdays] = varlist['Close']
mulCurves[:,1*numdays:2*numdays] = varlist['Open']
mulCurves[:,2*numdays:3*numdays] = varlist['High']
mulCurves[:,3*numdays:4*numdays] = varlist['Low']

# %% Cluster analysis

clusters = skc.DBSCAN(min_samples=50, metric=distance, eps=0.02, p=1).fit(mulCurves)
# %%
