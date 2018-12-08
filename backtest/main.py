import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import inv
from backtest import *
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from figures import *

plotly.tools.set_credentials_file(username='ummm', api_key="can't-let-you-know")


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





# First Backtesting:

# backtesting parameter table:
keys = ['Number of PCA Factors','Backtesting Frequency','Rolling Window','Efficient Portfolio']
values = [5,'Month','60 Months','Minimum Variance Portfolio (Short Sale Allowed)']
backtest_parameters_table = pd.DataFrame({'Parameters':keys,'Values':values})
backtest_parameters_table.to_csv('./backtest-report/data/backtest_parameters_table1.csv',index=False)


B = Backtesting(df,df_rf,'M',60,5)

B.backtest_unconstrained('MVP')


# figure 1
ewma=B.vol_ewma(60,0.94)
ewma_plot(ewma,'ewma')

# figure 2
p = B.price()

price_plot(p,'1_price')



# performance analysis:
# sharpe 
# sotino
# infomation ratio
# alpha
# beta
ratios = ['alpha','beta','Sharpe Ratio','IR','Sortino Ratio']
FC_perf = [B.alpha_beta()[0]['alpha'],B.alpha_beta()[0]['beta'],B.sharpe_ratio()['FC'],
           B.IR()['FC'], B.sortino_ratio()['FC']]
S_perf = [B.alpha_beta()[1]['alpha'],B.alpha_beta()[1]['beta'],B.sharpe_ratio()['Simply'],
           B.IR()['Simply'], B.sortino_ratio()['S']]
PA = pd.DataFrame({'Ratios':ratios, 'Factor Constrained':FC_perf, 'Simple Estimation':S_perf})
PA = PA[[ 'Ratios','Simple Estimation','Factor Constrained' ]]
PA = PA.round(4)
PA.to_csv('./backtest-report/data/performance_analysis.csv',index=False)
#*********************************************************





# Second Backtesting
B = Backtesting(df,df_rf,'M',60,5)
B.backtest_constrained()
p2 = B.price()
p2.plot()
price_plot2(p2,'2_price')




# performance analysis table:
ratios = ['alpha','beta','Sharpe Ratio','IR','Sortino Ratio']
FC_perf = [B.alpha_beta()[0]['alpha'],B.alpha_beta()[0]['beta'],B.sharpe_ratio()['FC'],
           B.IR()['FC'], B.sortino_ratio()['FC']]
S_perf = [B.alpha_beta()[1]['alpha'],B.alpha_beta()[1]['beta'],B.sharpe_ratio()['Simply'],
           B.IR()['Simply'], B.sortino_ratio()['S']]
PA = pd.DataFrame({'Ratios':ratios, 'Factor Constrained':FC_perf, 'Unconstrained':S_perf})
PA = PA[[ 'Ratios','Unconstrained','Factor Constrained' ]]
PA = PA.round(4)
PA.to_csv('./backtest-report/data/performance_analysis2.csv',index=False)




bt = B.backtest_ret
bt = bt.join(B.rf,how='left')
bt['FC'] += bt['RF']











