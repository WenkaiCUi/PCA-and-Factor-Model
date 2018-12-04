import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from numpy.linalg import inv
from backtest import *


df = pd.read_csv('../data/48_Industry_Portfolios_daily.csv',skiprows=9,nrows =  24328)
df2 = pd.read_csv(r'../data/44ind-mon.csv')

df_rf = pd.read_csv('../data/F-F_Research_Data_Factors_daily.CSV',skiprows=4,nrows = 24351)

# select some industries:
a = [i.strip() for i in list(df2)]
b = [i.strip() for i in list(df)]
(set(a)-set(b)) 
(set(b)-set(a))

intersect = list(set(a).intersection(set(b)))

df.columns = [i.strip() for i in df.columns.tolist()]
df = df[intersect]

# restructing df and df_rf
def reshape_df(df):
    # Date index
    df=df.rename(columns = {'Unnamed: 0':'Date'})
    df['Date'] = pd.to_datetime(df['Date'],format ='%Y%m%d')
    df.set_index('Date',inplace=True)
    # change normal return to log return
    df = np.log(df/100+1)
    return df

df = reshape_df(df)
df_rf = reshape_df(df_rf)        

#M= 60  fac=5   freq = M
#B = Backtesting(df,df_rf,'M',60,5)
#
#B.backtest_unconstrained('MVP')
#
## figure 1
#B.sample_sd()
#
## price:
#B.price().plot()
## Vol RiskMetrics estimate
#ewma=B.vol_ewma(60,0.94)
#ewma.plot()
#
#B.sharpe_ratio()

#*********************************************************
B = Backtesting(df,df_rf,'M',60,5)
B.backtest_constrained()
B.price().plot()
















