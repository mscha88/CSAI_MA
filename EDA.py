
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
from matplotlib.backends.backend_pdf import PdfPages


# In[2]:


data = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data = data.round(0)


# In[3]:


d1 = data.iloc[:1]
d1 = d1.T


# In[5]:


datat = data.T
data_ts = datat


# In[6]:


## filter based on financial account
datat_s = data_ts.loc[:, data_ts.columns.str.contains('Sales')]
datat_c = data_ts.loc[:, data_ts.columns.str.contains('Costs')]
datat_p = data_ts.loc[:, data_ts.columns.str.contains('Profit')]
datat_a = data_ts.loc[:, data_ts.columns.str.contains('Assets')]
datat_ll = data_ts.loc[:, data_ts.columns.str.contains('Long-term liabilities')]
datat_cl = data_ts.loc[:, data_ts.columns.str.contains('Current liabilities')]


# In[7]:


scaler = MinMaxScaler()
datat_sc = data_ts.values
datat_sc1 = scaler.fit(datat_sc)
datat_sc = scaler.transform(datat_sc)
dataset1 = datat_sc[0:60,]


# In[8]:


data_sc1 = pd.DataFrame(dataset1)


# In[9]:


data_sc1.describe()


# In[10]:


## display some time series

pp = PdfPages('line_plots.pdf')

for i in range (0,606, 1):
    fig = plt.figure()
    pyplot.plot(dataset1[:,i])
    pyplot.title("Normalized Financial Accounts by affiliate " + str(i+1))
    pyplot.xlabel("months")
    #pyplot.show()
    pp.savefig()
pp.close()


# In[11]:


scaler = MinMaxScaler()
datat_sc = datat_s.values
datat_sc1 = scaler.fit(datat_sc)
datat_sc = scaler.transform(datat_sc)


# In[12]:


series = d1


# In[13]:


## Autocorrelation and Partial Autocorrelation Plots

serie = (d1.astype(np.float64))
pyplot.figure()
lags = 60
    # acf
axis = pyplot.subplot(2, 1, 1)
plot_acf(serie, ax=axis, lags=lags, title='Autocorrelation function ' + str(1))
# pacf
axis = pyplot.subplot(2, 1, 2)
plot_pacf(serie, ax=axis, lags=lags, title='Partial Autocorrelation function ' + str(1))
# show plot
plt.subplots_adjust(hspace=0.5)
pyplot.show()


# In[14]:


data_l = datat.values
#print(data_l)
data = data[0:60]
data_l1 = data_l[1:,0].astype(np.float64)
#print(data_l1.shape)
#print(data_l1.dtype)
#print(data_l.shape[1])
#print(data_l.shape[0])


# In[15]:


# plot acf and pacf per time series before scaling
pp = PdfPages('autocorrelation & partial autocorrelation_plots_60.pdf')
for i in range(data_l.shape[1]):
    serie = (data_l[:,i].astype(np.float64))
    #print(serie)
    # plots
    pyplot.figure()
    lags = 60
    # acf
    axis = pyplot.subplot(2, 1, 1)
    plot_acf(serie, ax=axis, lags=lags, title='Autocorrelation function - time series ' + str(i+1))
    # pacf
    axis = pyplot.subplot(2, 1, 2)
    plot_pacf(serie, ax=axis, lags=lags, title='Partial Autocorrelation function - time series ' + str(i+1))
    # show plot
    plt.subplots_adjust(hspace=0.5)
    #pyplot.show()
    pp.savefig()
pp.close()


# In[16]:


scaler = MinMaxScaler()
data_sc = scaler.fit(data_l)
data_sc = scaler.transform(data_l)


# In[17]:


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


# In[18]:


i = 30
datat0 = datat.iloc[:, 0:i]


# In[19]:


#datat1 = datat0.values
datat_c = d1


# In[20]:


data_sc1 = pd.DataFrame(data = data_sc1)
#print(data_sc1)


# In[21]:


variables = data_sc1.columns
print(variables)


# In[22]:


df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
#print(df)


# In[19]:


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


# In[20]:


print(df.shape)


# In[27]:


c1 = 0 ## put into ranges!!
r1 = 0
r2 = 0
r3 = 0
r4 = 0
r5 = 0
r6 = 0
r7 = 0
r8 = 0
r9 = 0
r10 = 0
for i in range(0, df.shape[0]):
    for j in range(0, df.shape[0]):
        if df.loc[i, j] == 1:
            c1 += 1
        if df.loc[i, j] < 0.05:
            r1 += 1
        if df.loc[i, j] > 0.05000001 and df.loc[i, j] < 0.2:
            r2 += 1
        if df.loc[i, j] > 0.2000001 and df.loc[i, j] < 0.4:
            r3 += 1
        if df.loc[i, j] > 0.4000001:
            r4 += 1
print("total", df.shape[0]*df.shape[1])
print(c1)
print(r1)
print(r2)
print(r3)
print(r4)
print(r1+r2+r3+r4+c1)


# In[36]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ranges = ['>0.05', '0.05 to 0.2', '0.2 to 0.4', '<0.4']
values = [r1/(df.shape[0]*df.shape[1]),r2/(df.shape[0]*df.shape[1]),r3/(df.shape[0]*df.shape[1]),r4/(df.shape[0]*df.shape[1])]
ax.bar(ranges,values)
ax.set_title('Granger Causality Test Results in %')
plt.xlabel("p-value range")
#plt.ylabel("")
plt.show()


# In[24]:


c1 = 0 ## put into ranges!!
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
print("total", df.shape[0]*df.shape[1])
print(c1)
print(cbelow)
print(cabove)
print(cabove+cbelow+c1)


# In[47]:


pp = PdfPages('heatmaps.pdf')

for i in range(0, 606, 101):
    for j in range(0, 606, 101):
        corr1= df.values[i:i+50,j:j+50]
        plt.figure(figsize=(24, 16))
        ax = plt.axes()
        ax = sns.heatmap(np.round(corr1,1), ax = ax, linewidths=.2, annot=True, fmt='.2g')
        ax.set_title('Heatmap - Granger Causality Test Results ' + str(i) + ":" + str(i+50) + " versus " + str(j) + ":" + str(j+50))
        plt.xlabel("Time Series")
        plt.ylabel("Time Series")
        #plt.show()
        pp.savefig()
pp.close()


# In[44]:


pp = PdfPages('heatmaps.pdf')

for i in range(0, 606, 101):
    corr1= df.values[i:i+50,i:i+50]
    plt.figure(figsize=(24, 16))
    ax = plt.axes()
    ax = sns.heatmap(np.round(corr1,1), ax = ax, linewidths=.2, annot=True, fmt='.2g')
    ax.set_title('Heatmap - Granger Causality Test Results')
    plt.xlabel("Time Series")
    plt.ylabel("Time Series")
    #plt.show()
    pp.savefig()
pp.close()


# In[24]:


corr= df.values

plt.figure(figsize=(24, 16))
ax = plt.axes()
ax = sns.heatmap(corr, ax = ax)
ax.set_title('Heatmap - Granger Causality Test Results')
plt.xlabel("Time Series")
plt.ylabel("Time Series")
plt.show()


# In[23]:


## lag plot

plt.figure()
lag_plot(datat)


# In[24]:


print(datat.shape)


# In[25]:


i = 10
datatp = datat.iloc[:, 0:i]


# In[26]:


## scatter matrix

scatter_matrix(datatp, alpha=0.2, figsize=(6, 6), diagonal='kde');


# In[27]:


data1 = pd.read_excel("Data_v02.xlsx", sheet_name="Sheet1", skiprows=3, index_col=0)
data1 = data1.round(0)


# In[28]:


data1t = data1.T


# In[29]:


## filter based on financial account
data1t_s = data1t.loc[:, data1t.columns.str.contains('Sales')]
data1t_c = data1t.loc[:, data1t.columns.str.contains('Costs')]
data1t_p = data1t.loc[:, data1t.columns.str.contains('Profit')]
data1t_a = data1t.loc[:, data1t.columns.str.contains('Assets')]
data1t_ll = data1t.loc[:, data1t.columns.str.contains('Long-term liabilities')]
data1t_cl = data1t.loc[:, data1t.columns.str.contains('Current liabilities')]


# In[30]:


pp = PdfPages('decompose.pdf')

## display some time series
for i in range (1,606, 1):
    series = dataset1[:,i]
    result = seasonal_decompose(series, model='additive', freq=12)
    result.plot()
    pp.savefig()
pp.close()

