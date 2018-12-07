import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import inv
from backtest import *
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import copy

plotly.tools.set_credentials_file(username='cuiwk0320', api_key='1iAR7tJSOXMPBZgZhPzS')


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

# backtesting table:
keys = ['Number of PCA Factors','Backtesting Frequency','Rolling Window','Efficient Portfolio']
values = [5,'Month','60 Months','Minimum Variance Portfolio (Short Sale Allowed)']
backtest_parameters_table = pd.DataFrame({'Parameters':keys,'Values':values})
backtest_parameters_table.to_csv('./backtest-report/data/backtest_parameters_table1.csv',index=False)


B = Backtesting(df,df_rf,'M',60,5)

B.backtest_unconstrained('MVP')




# figure 1
ewma=B.vol_ewma(60,0.94)

mkt_trace = go.Scatter(
    x = ewma.index,
    y = ewma['Mkt-RF'],
    name = 'Market Portfolio',
    line = dict(
        color = '#163c6d',
        width = 2,
        dash = 'dash')
)

FC_trace = go.Scatter(
    x = ewma.index,
    y = ewma['FC'],
    name = 'Factor Constrained Portfolio',
    line = dict(
        color = '#8c0f07',
        width = 2.5)
)
    
S_trace = go.Scatter(
    x = ewma.index,
    y = ewma['Simply'],
    name = 'Unconstrained Portfolio',
    line = dict(
        color = '#163c6d',
        width = 2,
        dash = 'dot')
)
data = [mkt_trace,FC_trace,S_trace]
layout = dict(#title = '5-Year-Rolling RiskMetrics EWMA Volatility',
              plot_bgcolor='rgb(217,224,236)',
              autosize = True,
              margin = {
                                    "r": 0,
                                    "t": 0,
                                    "b": 30,
                                    "l": 25
                                  },
              annotations = [{
                      'text':'5-Year-Rolling EWMA Volatility',
                      'yref':'paper',
                      'y':1.12,
                      'xref':'paper',
                      'x':1,
                      'showarrow':False,
                      }],
              legend= {'bgcolor':('rgba(236, 240, 246,0.5)'),
                       'x':0.5,
                       'y':0.99
                       },
              yaxis = {'range':[0.05,0.35],
                       'gridcolor':'white',
                       'dtick':0.05},
              xaxis = {
                      'range':['1990-01-01', '2019-01-01'],
                      'gridcolor':'white',
                       "rangeselector": {"buttons": [
                                        {
                                          "count": 1,
                                          "label": "1Y",
                                          "step": "year",
                                          "stepmode": "backward"
                                        },
                                        {
                                          "count": 3,
                                          "label": "3Y",
                                          "step": "year",
                                          "stepmode": "backward"
                                        },
                                        {
                                          "count": 5,
                                          "label": "5Y",
                                          "step": "year"
                                        },
                                        {
                                          "count": 10,
                                          "label": "10Y",
                                          "step": "year",
                                          "stepmode": "backward"
                                        },
                                        {
                                          "label": "All",
                                          "step": "all"
                                        }
                                      ]}
                      }
              )

fig = dict(data=data, layout=layout)
py.plot(fig, filename='ewma')    

# figure 2
p = B.price()
mkt_trace = go.Scatter(
    x = p.index,
    y = p['Mkt-RF'],
    name = 'Market Portfolio',
    line = dict(
        color = '#163c6d',
        width = 2,
        dash = 'dash')
)

FC_trace = go.Scatter(
    x = p.index,
    y = p['FC'],
    name = 'Factor Constrained Portfolio',
    line = dict(
        color = '#8c0f07',
        width = 2.5)
)
    
S_trace = go.Scatter(
    x = p.index,
    y = p['Simply'],
    name = 'Unconstrained Portfolio',
    line = dict(
        color = '#163c6d',
        width = 2,
        dash = 'dot')
)    

data2 = [mkt_trace,FC_trace,S_trace]
layout2 = copy.deepcopy(layout)
layout2['annotations'][0]['text'] = 'Price of Portfolios'
layout2['yaxis']['range'] = [0,14]
layout2['xaxis']['range']=['1985-01-01', '2019-01-01']
layout2['yaxis']['dtick'] = 2
layout2['xaxis']['dtick'] = 'M60'


fig = dict(data=data2, layout=layout2)
py.plot(fig, filename='1_price',auto_open=True)    

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
#B = Backtesting(df,df_rf,'M',60,5)
#B.backtest_constrained()
#B.price().plot()
















