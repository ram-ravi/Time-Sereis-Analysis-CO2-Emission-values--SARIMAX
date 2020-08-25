# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 10:17:20 2020

@author: ramravi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import  the dataset
data=pd.read_csv('timeseries-dataset.csv')
data.info()
data.describe()
#we can convert teh datframe into a datetime dataframe adn cancel all the NaT values making the year as the index
dateparse=lambda x: pd.to_datetime(x, format='%Y%m', errors='coerce')
data=pd.read_csv('timeseries-dataset.csv', parse_dates=['YYYYMM'],
                 index_col=('YYYYMM'), date_parser=dateparse)
ts=data[pd.Series(pd.to_datetime(data.index, errors='coerce')).notnull().values]
ts.info()
#as we can see the emission value is in onject datatype, hence we convert it into numeric datatype
ts['Value']=pd.to_numeric(ts['Value'], errors='coerce')
ts.head()
ts.info()
ts.dropna(inplace=True)


#lets find out the emission value for each of the power generation:
energy_sorc= ts.groupby('Description')
energy_sorc.head()


fig, ax = plt.subplots()
for desc, group in energy_sorc:
    group.plot(y='Value', label=desc,ax = ax, title='Carbon Emissions per Energy Source', fontsize = 10)
    ax.set_xlabel('Time(Monthly)')
    ax.set_ylabel('Carbon Emissions in MMT')
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(20)
    ax.legend(fontsize = 5)
    
    
fig, axes= plt.subplots(3,3, figsize=(18,15))
for (desc, group), ax in zip(energy_sorc, axes.flatten()):
    group.plot(y='Value', ax=ax, title=desc, fontsize=18)
    ax.set_xlabel('Time(montnly')
    ax.set_ylabel('Value')
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)

#in the recent years natral gas consumption is increasing and the use of coal for power generation has been declining.
#bar chart of CO2 emission per energy source
co2_per_source=ts.groupby('Description')['Value'].sum().sort_values()
co2_per_source.index

cols=['Geothermal Energy', 'Non-Biomass Waste',
      'Petroleum Coke', 'Distillate Fuel',
      'Residual Fuel Oil', 'Petroleum', 'Natural Gas',
      'Coal','Total Emissions']

fig=plt.figure(figsize=(16,9))
x_label=cols
x_ticks=np.arange(len(cols))
plt.bar(x_ticks, co2_per_source, align='center',
        alpha=0.5)
fig.suptitle('CO2 Emission by Electric Power Sector', fontsize=25)
plt.xticks(x_ticks,x_label, rotation=70, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Carbon Emission in MMT', fontsize=20)
plt.show()

# From the graph, it is seen that the contribution of coal is significant followed by natural gas.

#natural gas emission analysis
emission=ts.iloc[:,1:]

emission=emission.groupby(['Description', pd.Grouper(freq='M')])['Value'].sum()
mte = emission['Natural Gas Electric Power Sector CO2 Emissions']

#test stationary

#The first thing we need to do is producing a plot of our time series dataset. From the plot, we will get an idea about the overall trend and seasonality of the series then, we will use a statistical method to assess the trend and seasonality of the dataset. After trend and seasonality are assessed if they are present in the dataset, they will be removed from the series to transform the nonstationary dataset into stationary and the residuals are further analyzed.

#A short summary about stationarity from Wikipedia: A stationary process is a stochastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance, if they are present, also do not change over time.

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

#graphically test stationary

plt.plot(mte)

#from the figure it is evident that the co2 emission has seasonal variation.


# Test Staionary property
#test stationary using dickey-fuller unit root test.
#this method is used to check the stationarity of the dataset. where we use the critical value(threshold value) and compare it with some statistical value at different confidence level.
#we state our null hypothesis that our dataset is non-stationary. If test statisitics < critical value, we can reject the null hypo.


def TestStationaryPlot(ts, plot_label=None):
    rol_mean=ts.rolling(window=12, center=False).mean()
    rol_std=ts.rolling(window=12, center=False).std()
    plt.plot(ts, color='blue', label='Original Data')
    plt.plot(rol_mean, color='red', label='Rolling mean')
    plt.plot(rol_std, color='black', label='Rolling std')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='best', fontsize=10)
    if plot_label is not None:
        plt.title('Rolling Mean & standard deviation(' + plot_label+ ')', fontsize=20)
    else:
        plt.title('Rolling mean & std deviation', fontsize=20)
    plt.show(block=True)
            
def TestStationaryAdfuller(ts, cutoff=0.01):
    ts_test=adfuller(ts, autolag='AIC')
    ts_test_output=pd.Series(ts_test[0:4], index=['Test Statistics', 'p-value','#Lags Used',                                           'Number of Observations Used'])
    for key, value in ts_test[4].items():
        ts_test_output['Critical value (%s)' %key]=value
    print(ts_test_output)
    
    if ts_test[1]<= cutoff:
        print('strong evidence against null hypo reject the null hypo.Data has no unit root, hence stationary')
    else:
        print('weak evidence against null hypo, non-stationary character')
        
        
#testing teh monthly emission time series:
 
TestStationaryPlot(mte,'unmodified data')


TestStationaryAdfuller(mte)


#Transforming the dataset into stationary dataset

#a. moving average

#we take the average of 'k' consecutive year depending on the freq of the time series.
#we will take the average of 12 months

moving_avg=mte.rolling(12).mean()
plt.plot(mte)
plt.plot(moving_avg, color='red')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Time (years)', fontsize=10)
plt.ylabel('CO2 emissions (MMT)', fontsize=10)
plt.title('Co2 emission from electric power generation')
plt.show()

mte_moving_avg_diff= mte- moving_avg
mte_moving_avg_diff.head(13)


mte_moving_avg_diff.dropna(inplace=True)

TestStationaryPlot(mte_moving_avg_diff, 'moving_average')

TestStationaryAdfuller(mte_moving_avg_diff)
#we can see that the test statistics value is smaller than the critical value at 99%, 95%, 90% confidence interval. Hence the dataset is stationary.

#B. Exponential weighted moving average

#**Another technique is to take the ‘weighted moving average’ where more recent values are given a higher weight. The popular method to assign the weights is using the exponential weighted moving average. In this technique, weights are assigned to all previous values with a decay factor.

mte_exp_weighted_avg=mte.ewm(halflife=12).mean()
plt.plot(mte)
plt.plot(mte_exp_weighted_avg)
plt.title('CO2 emission with respect to power generation')
plt.xlabel('time(year)')
plt.ylabel('co2 emission value(mmt)')
plt.show()

mte_exp_avg_diff= mte_exp_weighted_avg- mte
TestStationaryPlot(mte_exp_avg_diff, 'exponential weighted average')
TestStationaryAdfuller(mte_exp_avg_diff)

#Even, though the time series has lesser variation in mean and std dev compared to the original dataset, since the test statistics score is not smaller than teh critical values at various confidence level, failed to reject the null hypothesis.

#c. Eliminating trend and seasonality

#we are going to eliminate trend and seasonality using differencing. deff current value with the previous instant using first order differencing.

mte_first_diff= mte- mte.shift(1)
TestStationaryPlot(mte_first_diff.dropna(inplace=False), 'differencing')
TestStationaryAdfuller(mte_first_diff.dropna(inplace=False))

#The first difference improves the stationarity of the series significantly. Let us use also the seasonal difference to remove the seasonality of the data and see how that impacts stationarity of the data.

mte_seasonal_diff=mte-mte.shift(12)

TestStationaryPlot(mte_seasonal_diff.dropna(inplace=False),'seasonal difference')
TestStationaryAdfuller(mte_seasonal_diff.dropna(inplace=False))

#Compared to the original data the seasonal difference also improves the stationarity of the series. The next step is to take the first difference of the seasonal difference.

mte_seasonal_first_diff=mte_seasonal_diff-mte_seasonal_diff.shift(1)
TestStationaryPlot(mte_seasonal_first_diff.dropna(inplace=False),'diff of seasonal diff')
TestStationaryAdfuller(mte_seasonal_first_diff.dropna(inplace=False))

#Now, if we look the Test Statistic and the p-value, taking the seasonal first difference has made our the time series dataset stationary. This differencing procedure could be repeated for the log values, but it didn’t make the dataset any more stationary.

#d. Eliminating trend and seasonality using decomposing

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(mte)

trend= decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid


plt.subplot(411)
plt.plot(mte, label='original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='residual')
plt.legend(loc='best')


#lets see the staionarity of residual :

mte_decompose=residual
mte_decompose.dropna(inplace=True)
TestStationaryPlot(mte_decompose,'decomposing')
TestStationaryAdfuller(mte_decompose)

#SARIMAX using optimal paramters

fig= plt.figure(figsize=(12,18))
ax1=fig.add_subplot(211)
fig= sm.graphics.tsa.plot_acf(mte_seasonal_first_diff.iloc[13:], lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(mte_seasonal_first_diff.iloc[13:], lags=40, ax=ax2)

#grid search for optimal paramteters:
import itertools
p=d=q=range(0,2)
#p,d,q and s to take any value between 0 and 2
pdq=list(itertools.product(p,d,q))
#generate all the different combinations of p,d,q triplets
pdq_x_QDQs=[(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]

print('Example of seasonal ARIMA paramter combinations for seasonal ARIMA...')
print('SARIMAX: {} x {}' .format(pdq[1], pdq_x_QDQs[1]))
print('SARIMAX: {} x {}' .format(pdq[2], pdq_x_QDQs[2]))

for param in pdq:
    for seasonal_param in pdq_x_QDQs:
        try:
            mod= sm.tsa.statespace.SARIMAX(mte,
                                           order=param,
                                           seasonal_order=seasonal_param,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
            result=mod.fit()
            print('ARIMA{} x {}-AIC{}' .format(param, param_seasonal, result.aic))
        except:
                continue
            


mod=sm.tsa.statespace.SARIMAX(mte,
                              order=(1,1,1),
                              seasonal_order=(0,1,1,12),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
results=mod.fit()
print(results.summary())


results.resid.plot()
print(results.resid.describe())

results.resid.plot(kind='kde')
results.plot_diagnostics(figsize=(15,12))
#from the results, we can infer that, the results areof residual error follows a normal distribution.
#In the top right plot, the red KDE line follows closely with the N(0,1) line. Where, N(0,1) is the standard notation for a normal distribution with mean 0 and standard deviation of 1. This is a good indication that the residuals are normally distributed. The forecast errors deviate somewhat from the straight line, indicating that the normal distribution is not a perfect model for the distribution of forecast errors, but it is not unreasonable.
#The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) follows the linear trend of the samples taken from a standard normal distribution. Again, this is a strong indication that the residuals are normally distributed.
#The residuals over time (top left plot) don't display any obvious seasonality and appear to be white noise. This is confirmed by the autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have low correlation with lagged versions of itself.




#validation prediction:
pred= results.get_prediction(start=480, end=523, dynamic=False)
pred_ci=pred.conf_int()
pred_ci.head()

#The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.

#plotting the forecast models:
ax=mte['1973':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='one-step ahead forecast', alpha=0.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1], color='r', alpha=0.5)
ax.set_xlabel('Time(years)')
ax.set_ylabel('NG CO2 Emissions')
plt.legend()
plt.show()



#Accuracy using MSE:

    
mte_forecast= pred.predicted_mean
mte_truth=mte['2013-01-31':]

#compute the mean sqaured error:
mse=((mte_forecast-mte_truth)**2).mean()
print('the mean sqaured error (mse) of the forecast is {}' .format(round(mse, 2)))
print('the root mean sqaured error(rmse) {:.4f}' .format(np.sqrt(sum((mte_forecast-mte_truth)**2)/len(mte_forecast))))


mte_pred_concat=pd.concat([mte_truth,mte_forecast])


#our aim is to dynamically forecast the co2 emission values.
pred_dynamic=results.get_prediction(start=pd.to_datetime('2013-01-31'), dynamic=True, full_results=True)
pred_dynamic_ci= pred_dynamic.conf_int()

#plotting the observation
ax=mte['1973':].plot(label='observed', figsize=(20,15))
pred_dynamic.predicted_mean.plot(label='Dynamic forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:,0],
                pred_dynamic_ci.iloc[:,1],
                color='r',
                alpha=0.3)
ax.fill_betweenx(ax.get_ylim(),
                pd.to_datetime('2013-01-31'),
                mte.index[-1],
                alpha=0.1, zorder=-1)

ax.set_xlabel('Time(years)')
ax.set_ylabel('CO2 emission')

plt.legend()
plt.show()

#extract teh predicted ad true value in time series
mte_forecast= pred_dynamic.predicted_mean
mte_original=mte['2013-01-31':]

mse=((mte_forecast-mte_original)**2).mean()
print('teh mse value for the forecast is {}' .format(round(mse,2)))
print('rmse for the forecast is  {:.4f}' .format(np.sqrt(sum((mte_forecast-mte_original)**2)/len(mte_forecast))))

#forecast
forecast= results.get_forecast(steps=120)
#get the confidence intervals of forecast
forecast_ci=forecast.conf_int()
forecast_ci.head()

ax=mte.plot(label='observed', figsize=(20,15))
forecast.predicted_mean.plot(ax=ax, label='forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:,0],
                forecast_ci.iloc[:,1],
                color='g',
                alpha=0.4)
ax.set_xlabel('Time(year)')
ax.set_ylabel('CO2 emission')
plt.title('forecast prediction', fontsize=(10))
plt.legend()
plt.show()

#Both the forecast and associated confidence interval that we have generated can now be used to further explore and understand the time series. The forecast shows that the CO2 emission from natural gas power generation is expected to continue increasing.
    








































































































#using these test, we can clearly see the variation in std and mean.Hence it is not stationary. Further, our test statistics are greater than the critical value at 99%, 95%, 90% confidence interval.
#hence cant reject the null hypothesis.

Transform the dataset to stationary






















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    