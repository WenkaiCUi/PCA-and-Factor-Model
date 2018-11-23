import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

df = pd.read_csv('./48_Industry_Portfolios_daily.csv',skiprows=9,nrows =  24328)
df=df.rename(columns = {'Unnamed: 0':'Date'})
df['Date'] = pd.to_datetime(df['Date'],format ='%Y%m%d')
df.set_index('Date',inplace=True)

# select data after 1980
df=df.loc['1980':]


# change normal return to log return
df = np.log(df/100+1)

# daily return to weekly return
df_Week = df.resample('W').sum()

# backtesting:

# set 3 yr rolling window
M = 52*3  
k = 10  # number of factors
i = 0


df_train = df.iloc[i:i+M,]




