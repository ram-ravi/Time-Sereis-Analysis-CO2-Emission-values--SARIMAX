### Time-Sereis-Analysis-CO2-Emission-values--SARIMAX(Seasonal ARIMA)

#### Time Series dataset:
I use a public dataset of monthly carbon dioxide emissions from electricity generation available at the Energy Information Administration and Jason McNeill. The dataset includes CO2 emissions from each energy 
resource starting January 1973 to July 2016.

lets find out the emission value for each of the power generation:<br>
<img src="https://user-images.githubusercontent.com/64869288/91193160-a3cae280-e6ab-11ea-946d-2fbb97cb4900.png" width="750" height="350">

#in the recent years natral gas consumption is increasing and the use of coal for power generation has been declining.
#bar chart of CO2 emission per energy source


From the graph, it is seen that the contribution of coal is significant followed by natural gas.


#### natural gas emission analysis



#### test stationary

The first thing we need to do is producing a plot of our time series dataset. From the plot, we will get an idea about the overall trend and seasonality of the series then, we will use a statistical method to assess the trend and seasonality of the dataset. After trend and seasonality are assessed if they are present in the dataset, they will be removed from the series to transform the nonstationary dataset into stationary and the residuals are further analyzed.

A short summary about stationarity from Wikipedia: A stationary process is a stochastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance, if they are present, also do not change over time.

#### graphically test stationary

##### from the figure it is evident that the co2 emission has seasonal variation.



#### Test Staionary property
test stationary using dickey-fuller unit root test.
this method is used to check the stationarity of the dataset. where we use the critical value(threshold value) and compare it with some statistical value at different confidence level.
we state our null hypothesis that our dataset is non-stationary. If test statisitics < critical value, we can reject the null hypo.


#### testing teh monthly emission time series:


#Transforming the dataset into station  ary dataset

#a. moving average

#we take the average of 'k' consecutive year depending on the freq of the time series.
#we will take the average of 12 months



we can see that the test statistics value is smaller than the critical value at 99%, 95%, 90% confidence interval. Hence the dataset is stationary.

#### B. Exponential weighted moving average

**Another technique is to take the ‘weighted moving average’ where more recent values are given a higher weight. The popular method to assign the weights is using the exponential weighted moving average. In this technique, weights are assigned to all previous values with a decay factor.

#Even, though the time series has lesser variation in mean and std dev compared to the original dataset, since the test statistics score is not smaller than teh critical values at various confidence level, failed to reject the null hypothesis.

#### c. Eliminating trend and seasonality

The first difference improves the stationarity of the series significantly. Let us use also the seasonal difference to remove the seasonality of the data and see how that impacts stationarity of the data.


Compared to the original data the seasonal difference also improves the stationarity of the series. The next step is to take the first difference of the seasonal difference.

Now, if we look the Test Statistic and the p-value, taking the seasonal first difference has made our the time series dataset stationary. This differencing procedure could be repeated for the log values, but it didn’t make the dataset any more stationary.

#### d. Eliminating trend and seasonality using decomposing




#### lets see the staionarity of residual :



#### SARIMAX using optimal paramters

From the results, we can infer that, the results areof residual error follows a normal distribution.
In the top right plot, the red KDE line follows closely with the N(0,1) line. Where, N(0,1) is the standard notation for a normal distribution with mean 0 and standard deviation of 1. This is a good indication that the residuals are normally distributed. The forecast errors deviate somewhat from the straight line, indicating that the normal distribution is not a perfect model for the distribution of forecast errors, but it is not unreasonable.
The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) follows the linear trend of the samples taken from a standard normal distribution. Again, this is a strong indication that the residuals are normally distributed.
The residuals over time (top left plot) don't display any obvious seasonality and appear to be white noise. This is confirmed by the autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have low correlation with lagged versions of itself.






#### validation prediction:


The dynamic=False argument ensures that we produce one-step ahead forecasts, meaning that forecasts at each point are generated using the full history up to that point.



#### plotting the forecast models:


#### Accuracy using MSE:

#### our aim is to dynamically forecast the co2 emission values.








#### Conclusion:
Both the forecast and associated confidence interval that we have generated can now be used to further explore and understand the time series. The forecast shows that the CO2 emission from natural gas power generation is expected to continue increasing.
























































































