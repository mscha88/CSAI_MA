
# coding: utf-8

# In[1]:


import pandas as pd
from pandas.plotting import lag_plot
from pandas.plotting import scatter_matrix
from pandas import Series
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import split
from numpy import array
np.random.seed(0)
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns


# In[2]:


data = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data = data.round(0)
data.describe()


# In[3]:


d1 = data.iloc[:1]
d1 = d1.T


# In[4]:


datat = data.T
data_ts = datat


# In[5]:


## filter based on financial account
datat_s = data_ts.loc[:, data_ts.columns.str.contains('Sales')]
datat_c = data_ts.loc[:, data_ts.columns.str.contains('Costs')]
datat_p = data_ts.loc[:, data_ts.columns.str.contains('Profit')]
datat_a = data_ts.loc[:, data_ts.columns.str.contains('Assets')]
datat_ll = data_ts.loc[:, data_ts.columns.str.contains('Long-term liabilities')]
datat_cl = data_ts.loc[:, data_ts.columns.str.contains('Current liabilities')]


# In[6]:


scaler = MinMaxScaler()
datat_sc = data_ts.values
datat_sc1 = scaler.fit(datat_sc)
datat_sc = scaler.transform(datat_sc)
dataset1 = datat_sc[0:60,]


# In[10]:


data_sc1 = pd.DataFrame(dataset1)


# In[11]:


data_sc1.describe()


# In[28]:


## display some time series
for i in range (1,606, 101):
    pyplot.plot(dataset1[:,i])
    pyplot.show()


# In[13]:


scaler = MinMaxScaler()
datat_sc = datat_s.values
datat_sc1 = scaler.fit(datat_sc)
datat_sc = scaler.transform(datat_sc)


# In[40]:


series = d1


# In[36]:


## Autocorrelation and Partial Autocorrelation Plots

serie = (d1.astype(np.float64))
pyplot.figure()
lags = 24
    # acf
axis = pyplot.subplot(2, 1, 1)
plot_acf(serie, ax=axis, lags=lags)
# pacf
axis = pyplot.subplot(2, 1, 2)
plot_pacf(serie, ax=axis, lags=lags)
# show plot
plt.subplots_adjust(hspace=0.5)
pyplot.show()


# In[82]:


data_l = datat.values
#print(data_l)
data = data[0:60]
data_l1 = data_l[1:,0].astype(np.float64)
#print(data_l1.shape)
#print(data_l1.dtype)
#print(data_l.shape[1])
#print(data_l.shape[0])


# In[78]:


# plot acf and pacf per time series before scaling

for i in range(data_l.shape[1]):
    serie = (data_l[:,i].astype(np.float64))
    #print(serie)
    # plots
    pyplot.figure()
    lags = 68
    # acf
    axis = pyplot.subplot(2, 1, 1)
    plot_acf(serie, ax=axis, lags=lags)
    # pacf
    axis = pyplot.subplot(2, 1, 2)
    plot_pacf(serie, ax=axis, lags=lags)
    # show plot
    plt.subplots_adjust(hspace=0.5)
    pyplot.show()


# In[43]:


scaler = MinMaxScaler()
data_sc = scaler.fit(data_l)
data_sc = scaler.transform(data_l)


# In[44]:


# plot acf and pacf per time series after scaling

for i in range(data_sc.shape[1]):
    serie = (data_sc[:,i].astype(np.float64))
    #print(serie)
    # plots
    pyplot.figure()
    lags = 68
    # acf
    axis = pyplot.subplot(2, 1, 1)
    plot_acf(serie, ax=axis, lags=lags)
    # pacf
    axis = pyplot.subplot(2, 1, 2)
    plot_pacf(serie, ax=axis, lags=lags)
    # show plot
    plt.subplots_adjust(hspace=0.5)
    pyplot.show()


# In[ ]:


i = 30
datat0 = datat.iloc[:, 0:i]


# In[42]:


#datat1 = datat0.values
datat_c = d1


# In[45]:


data_sc1 = pd.DataFrame(data = data_sc1)
#print(data_sc1)


# In[46]:


variables = data_sc1.columns
print(variables)


# In[47]:


df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
#print(df)


# In[48]:


## Granger Causality tests

maxlag=12
test = 'ssr_chi2test'

verbose=False

for c in df.columns:
    for r in df.index:
        test_result = grangercausalitytests(data_sc1[[r, c]], maxlag=maxlag, verbose=False)
        p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
        if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
        min_p_value = np.min(p_values)
        df.loc[r, c] = min_p_value


# In[83]:


c1 = 0
cbelow = 0
cabove = 0
for i in range(0, df.shape[0]):
    for j in range(0, df.shape[0]):
        if df.loc[i, j] == 1:
            c1 += 1
        if df.loc[i, j] > 0.05:
            cabove += 1
        if df.loc[i, j] < 0.05:
            cbelow += 1
#print("total", df.shape[0]*df.shape[1])
#print(c1)
#print(cbelow)
#print(cabove)
#print(cabove+cbelow+c1)


# In[52]:


## significant correlation 

print("below", round(cbelow/(df.shape[0]*df.shape[1]),2))
print("above", round(cabove/(df.shape[0]*df.shape[1]),2))


# In[174]:


corr= df.values

plt.figure(figsize=(24, 16))
ax = plt.axes()
ax = sns.heatmap(corr, ax = ax)
ax.set_title('Heatmap - Granger Causality Test Results')
plt.xlabel("Time Series")
plt.ylabel("Time Series")
plt.show()


# In[85]:


## lag plot

plt.figure()
lag_plot(datat)


# In[65]:


print(datat.shape)


# In[66]:


i = 10
datatp = datat.iloc[:, 0:i]


# In[67]:


## scatter matrix

scatter_matrix(datatp, alpha=0.2, figsize=(6, 6), diagonal='kde');


# In[86]:


data1 = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data1 = data1.round(0)


# In[72]:


data1t = data1.T


# In[74]:


## filter based on financial account
data1t_s = data1t.loc[:, data1t.columns.str.contains('Sales')]
data1t_c = data1t.loc[:, data1t.columns.str.contains('Costs')]
data1t_p = data1t.loc[:, data1t.columns.str.contains('Profit')]
data1t_a = data1t.loc[:, data1t.columns.str.contains('Assets')]
data1t_ll = data1t.loc[:, data1t.columns.str.contains('Long-term liabilities')]
data1t_cl = data1t.loc[:, data1t.columns.str.contains('Current liabilities')]


# In[76]:


## sales accounts

freq = 12
for i in data1t_s:
    series = data1t_s[i]
    result = seasonal_decompose(series, model='additive', freq=freq)
    result.plot()
    pyplot.show()

