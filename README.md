# Adani-Enterprise-stock-exchange

Start by importing the necessary libraries such as Pandas, Numpy and Matplotlib.
Use the Pandas library to read the dataset into a dataframe.
Use the head() and info() functions to get a quick overview of the dataset.
Use the describe() function to get some basic statistics of the numerical columns.
Use visualizations such as histograms, box plots and scatter plots to understand the distribution of the data and detect any outliers.

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from pmdarima.arima import OCSBTest 
from statsmodels.tsa.arima_model import ARIMA 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()

plt.plot(df_train,color = "black")
plt.plot(df_test,color = "red")
plt.title("Train/Test split for Adani stock price")
plt.ylabel("Close")
plt.xlabel("Date")
sns.set()
plt.show()

sgt.plot_acf(df_train,lags=40,zero=False)
sgt.plot_pacf(df_train,lags=40,zero=False, method=('ols'))

from pmdarima.arima import auto_arima
model = auto_arima(df_train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(df_train)

plt.plot(df_train, color="blue",label="Original Close")
plt.plot(df_test, color="green",label="Test")
plt.plot(forecast, color="red",label="forecast")
plt.title("Adani Enterprise stock prise")
plt.ylabel("Closing")
plt.xlabel("Date")
plt.legend(['train','test','forecast'])
