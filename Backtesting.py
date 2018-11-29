import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from numpy.linalg import inv

df = pd.read_csv('./48_Industry_Portfolios_daily.csv',skiprows=9,nrows =  24328)
df2 = pd.read_csv(r'C:\Users\cui_w\Desktop\2017CourseMaterial\825\HW\44ind-mon.csv')

a = [i.strip() for i in list(df2)]
b = [i.strip() for i in list(df)]
(set(a)-set(b)) 
(set(b)-set(a))

intersect = list(set(a).intersection(set(b)))

df.columns = [i.strip() for i in df.columns.tolist()]
df = df[intersect]


df=df.rename(columns = {'Unnamed: 0':'Date'})
df['Date'] = pd.to_datetime(df['Date'],format ='%Y%m%d')
df.set_index('Date',inplace=True)

# select data after 1980
df=df.loc['1980':]

# change normal return to log return
df = np.log(df/100+1)

# daily return to weekly return
class Backtesting():
    def __init__(self,frequency,window,factors):
        self.__port_ret = None
        self.frequency = frequency
        self.window = window
        self.factors = factors
        
        
    def MVP(self):
        df_week = df.resample(self.frequency).sum()
        
        # backtesting:
        
        # set 3 yr rolling window
        M = self.window
        k = self.factors  # number of factors
        
        
        # first weight calculate from first 3 yr, apply to 3yr + 1 th week
        w_S = np.zeros((df_week.shape[0]-M,df_week.shape[1]))
        w_FC = np.zeros((df_week.shape[0]-M,df_week.shape[1]))
        
        # iterator
        for i in range(w_S.shape[0]):
            print(i)
            # no need to standardize
            df_train = df_week.iloc[i:i+M,]
            pca = PCA()
            pca.fit(df_train)
            P = np.transpose(pca.components_)
            lambd = np.diag(pca.explained_variance_)
            Omega = np.cov(df_train.T)
            B = P @ np.sqrt(lambd)
            
            # truncate B
            beta_hat = B[:,:k]
            D_hat = Omega - np.matmul(beta_hat,beta_hat.T)
            D_hat = np.diag(np.diagonal(D_hat))
            
            F_hat = np.zeros((M,k))
            for j in range(M):
                F_hat[j,:] = inv(beta_hat.T @ inv(D_hat) @ beta_hat) @ (beta_hat.T @ inv(D_hat) @ df_train.iloc[j,:].values)
            
            # residuals
            resid = df_train - F_hat @ beta_hat.T
            D = np.diag(np.diagonal(np.cov(resid.T)))
            Omega_F = np.cov(F_hat.T)
            # Factor constrained Covariance Matrix
            Omega_FC = beta_hat @ Omega_F @ beta_hat.T + D
            
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
        port_ret = np.zeros((df_week.shape[0]-M,2))
        for i in range(port_ret.shape[0]):
            df_test = df_week.iloc[i+M,:]
            port_ret[i,0] = w_FC[i,:] @ df_test
            port_ret[i,1] = w_S[i,:] @ df_test
        
        port_ret = pd.DataFrame(port_ret)
        port_ret.columns = ['FC','Simply']
        
        self.__port_ret = port_ret
        return port_ret
    
    def sample_sd(self):
        if self.__port_ret is None:
            print('Run backtesting first')
            return None
        else:
            return {'freq':self.frequency, 'M':self.window, 'fac':self.factors,\
                    'FC':np.std(self.__port_ret.iloc[:,0]),\
                    'Simply':np.std(self.__port_ret.iloc[:,1])}
            
    

freqs = ['W','M']
windows_w = [52*3,52*5,52*8]
windows_m = [12*3,12*5,12*8]
facs = [3,5,7,9]

lst =[]
for freq in freqs:
    if freq =='W':
        for M in windows_w:
            for fac in facs:
                B = Backtesting(freq,M,fac)
                B.MVP()
                lst = lst+ [B.sample_sd()]
                
    elif freq =='M':
        for M in windows_m:
            for fac in facs:
                B = Backtesting(freq,M,fac)
                B.MVP()
                lst = lst+ [B.sample_sd()]
                
lst = pd.DataFrame(lst)


# M= 60  fac=5   freq = M










