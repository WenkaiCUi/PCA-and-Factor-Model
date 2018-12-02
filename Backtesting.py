import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from numpy.linalg import inv

df = pd.read_csv('./48_Industry_Portfolios_daily.csv',skiprows=9,nrows =  24328)
df2 = pd.read_csv(r'C:\Users\cui_w\Desktop\2017CourseMaterial\825\HW\44ind-mon.csv')

df_rf = pd.read_csv('./F-F_Research_Data_Factors_daily.CSV',skiprows=4,nrows = 24351)

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


class Backtesting():
    def __init__(self,df,df_rf,frequency,window,factors,start='1980',end='2017'):
        self.frequency = frequency
        self.window = window     # rolling window to generate PCA factors
        self.factors = factors
        
        self.rf = df_rf['RF'].loc[start:end].resample(self.frequency).sum()
        self.mkt_rf =  df_rf['Mkt-RF'].loc[start:end].resample(self.frequency).sum()
        
        self.port_exret = df.loc[start:end].resample(self.frequency).sum()
        self.port_exret = self.port_exret.sub(self.rf,axis=0)
            
        self.backtest_ret = None   # generated after backtesting

        
    
    def _factor_cov(self,df,k):
        M = self.window
        pca = PCA()
        pca.fit(df)
        P = np.transpose(pca.components_)
        lambd = np.diag(pca.explained_variance_)
        Omega = np.cov(df.T)
        B = P @ np.sqrt(lambd)
        
        # truncate B
        beta_hat = B[:,:k]
        D_hat = Omega - np.matmul(beta_hat,beta_hat.T)
        D_hat = np.diag(np.diagonal(D_hat))
        
        F_hat = np.zeros((M,k))
        for j in range(M):
            F_hat[j,:] = inv(beta_hat.T @ inv(D_hat) @ beta_hat) @ (beta_hat.T @ inv(D_hat) @ df.iloc[j,:].values)
        
        # residuals
        resid = df - F_hat @ beta_hat.T
        D = np.diag(np.diagonal(np.cov(resid.T)))
        Omega_F = np.cov(F_hat.T)
        # Factor constrained Covariance Matrix
        Omega_FC = beta_hat @ Omega_F @ beta_hat.T + D
        return Omega_FC

        
    def MVP(self):
        # backtesting:       
        # set 3 yr rolling window
        M = self.window
        k = self.factors  # number of factors
              
        # first weight calculate from first 3 yr, apply to 3yr + 1 th week
        w_S = np.zeros((self.port_exret.shape[0]-M,self.port_exret.shape[1]))
        w_FC = np.zeros((self.port_exret.shape[0]-M,self.port_exret.shape[1]))
        
        # generate weights
        for i in range(w_S.shape[0]):
            print(i)
            # no need to standardize
            df_train = self.port_exret.iloc[i:i+M,]
            Omega_FC = self._factor_cov(df_train,k)
            
            # simple estimate covariance matrix:
            Omega_S = np.cov(df_train.T)
            
            # Minimum Variance Portfolio:
            ivec = np.ones(Omega_FC.shape[0])
            _w_FC = inv(Omega_FC) @ ivec
            _w_FC = _w_FC/_w_FC.sum()
            _w_S = inv(Omega_S) @ ivec
            _w_S = _w_S/_w_S.sum()
            
            w_FC[i,:] = _w_FC
            w_S[i,:] = _w_S
        
        # backtesting 
        backtest_ret = np.zeros((self.port_exret.shape[0]-M,2))
        for i in range(backtest_ret.shape[0]):
            df_test = self.port_exret.iloc[i+M,:]
            backtest_ret[i,0] = w_FC[i,:] @ df_test
            backtest_ret[i,1] = w_S[i,:] @ df_test
        
        backtest_ret = pd.DataFrame(backtest_ret)
        backtest_ret.columns = ['FC','Simply']
        
        backtest_ret.index = self.port_exret.iloc[M:,].index
        self.backtest_ret = backtest_ret
        return backtest_ret
    
    
    def sample_sd(self):
        if self.backtest_ret is None:
            print('Run backtesting first')
            return None
        else:
            return {'freq':self.frequency, 'M':self.window, 'fac':self.factors,\
                    'FC':np.std(self.backtest_ret.iloc[:,0]),\
                    'Simply':np.std(self.backtest_ret.iloc[:,1])}

             
    def ewma(self,M,lambd):
        # calculate EWMA of portfolio returns

        # first update
        sigma = np.zeros((self.backtest_ret.shape[0]-M,2))
        sigma_init = self.backtest_ret.iloc[:M,:].apply(np.std,axis=0)
        mu_init = self.backtest_ret.iloc[:M,:].apply(np.mean,axis=0)
        sigma[0,:] = lambd * sigma_init + (1-lambd)* (self.backtest_ret.iloc[M,]-mu_init)**2
        
        # 1-3 2-4 3-5   m=3 n=5
        for i in range(1,self.backtest_ret.shape[0]-M):
            mu = self.backtest_ret.iloc[i:M+i,:].apply(np.mean,axis=0)
            sigma[i,:] = lambd * sigma[i-1,:] + (1-lambd)* (self.backtest_ret.iloc[i+M,]-mu)**2
        
        sigma = pd.DataFrame(sigma,index=self.backtest_ret.iloc[M:,].index,columns =['FC','Simply'])

        return sigma
    
    def sharpe_ratio(self):
        mu = self.backtest_ret.apply(np.mean,axis=0)*12
        sd = self.backtest_ret.apply(np.std,axis=0)*np.sqrt(12)
        return mu/sd
    
    
        
        

#freqs = ['W','M']
#windows_w = [52*3,52*5,52*8]
#windows_m = [12*3,12*5,12*8]
#facs = [3,5,7,9]
#
#lst =[]
#for freq in freqs:
#    if freq =='W':
#        for M in windows_w:
#            for fac in facs:
#                B = Backtesting(freq,M,fac)
#                B.MVP()
#                lst = lst+ [B.sample_sd()]
#                
#    elif freq =='M':
#        for M in windows_m:
#            for fac in facs:
#                B = Backtesting(freq,M,fac)
#                B.MVP()
#                lst = lst+ [B.sample_sd()]
#  
#lst = pd.DataFrame(lst)


# M= 60  fac=5   freq = M
B = Backtesting(df,df_rf,'M',60,5)
B.MVP()

# figure 1
B.sample_sd()
# RiskMetrics estimate
ewma=B.ewma(60,0.94)


# test
#rf = df_rf['RF'].loc[start:end].resample('M').sum()
#mkt_rf =  df_rf['Mkt-RF'].loc[start:end].resample('M').sum()
#
#data = df.loc[start:end].resample('M').sum()
#data2 = data.sub(rf,axis=0)
#
#print(data2.iloc[4,2])
#print(data.iloc[4,2] -  rf.iloc[4])

