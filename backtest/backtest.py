import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.linalg import inv
import cvxopt as opt
from cvxopt import blas, solvers
from sklearn import linear_model
import math
    
class Backtesting():
    def __init__(self,df,df_rf,frequency,window,factors,start='1980',end='2017'):
        # backtest attributes
        self.backtest_ret = None   # generated after backtesting

        # PCA factor model attributes
        self.frequency = frequency    #data frequency
        self.N = {'M':12,'W':252}[self.frequency]   # annulization facotr
        self.window = window     # rolling window to generate PCA factors
        self.factors = factors   # factors kept in factor model
        
        # data
        self.rf = df_rf['RF'].loc[start:end].resample(self.frequency).sum()
        self.mkt_rf =  df_rf['Mkt-RF'].loc[start:end].resample(self.frequency).sum()
        self.port_exret = df.loc[start:end].resample(self.frequency).sum()
        self.port_exret = self.port_exret.sub(self.rf,axis=0)
            

        
    
    def _factor_cov(self,df):
        # input: training data
        # output: PCA factor constrained covariance matrix
        k = self.factors
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

    def _get_ret_by_weight(self,w_FC,w_S):
        M = self.window
        backtest_ret = np.zeros((self.port_exret.shape[0]-M,2))
        for i in range(backtest_ret.shape[0]):
            df_test = self.port_exret.iloc[i+M,:]
            backtest_ret[i,0] = w_FC[i,:] @ df_test
            backtest_ret[i,1] = w_S[i,:] @ df_test
        
        backtest_ret = pd.DataFrame(backtest_ret)
        backtest_ret.columns = ['FC','Simply']
        
        backtest_ret.index = self.port_exret.iloc[M:,].index
        
        # synthesize mkt return
        backtest_ret = backtest_ret.join(self.mkt_rf,how='left')
        return backtest_ret


    def backtest_unconstrained(self, port_type):
        self.backtest_ret = None

        def mvp(Omega):
            # analytical solution for MVP
            ivec = np.ones(Omega.shape[0])
            _w = inv(Omega) @ ivec
            _w = _w/_w.sum()
            return _w

        def tangency(Omega,mu):
            # analytical solution for tangency portfolios
            w = inv(Omega) @ mu
            w = w/w.sum()
            return w

        # backtesting:       
        M = self.window
              
        # first weight calculate from first 3 yr, apply to 3yr + 1 th week
        w_S = np.zeros((self.port_exret.shape[0]-M,self.port_exret.shape[1]))
        w_FC = np.zeros((self.port_exret.shape[0]-M,self.port_exret.shape[1]))
        
        # generate weights
        for i in range(w_S.shape[0]):
            print(i)
            # no need to standardize
            df_train = self.port_exret.iloc[i:i+M,]
            Omega_FC = self._factor_cov(df_train)
            
            # simple estimate covariance matrix:
            Omega_S = np.cov(df_train.T)
            
            # Minimum Variance Portfolio:
            if port_type == 'MVP':
                w_FC[i,:] = mvp(Omega_FC)
                w_S[i,:] = mvp(Omega_S)

            elif port_type == 'tangency':
                mu = df_train.apply(np.mean,axis=0)
                w_FC[i,:] = tangency(Omega_FC,mu)
                w_S[i,:] = tangency(Omega_S,mu)


        # backtesting 
        backtest_ret = self._get_ret_by_weight(w_FC,w_S)
        self.backtest_ret = backtest_ret

        return backtest_ret
    
    def backtest_constrained(self):

        def tang_contrained(Omega,mu):
            # numerical solve tangent portfolio
            n = Omega.shape[0]
            P = opt.matrix(Omega,tc='d')
            q = opt.matrix(0.0, (n ,1))
            G = -opt.matrix(np.eye(n))
            h = opt.matrix(0.0, (n ,1))
            A = opt.matrix(mu,(1,n))
            b = opt.matrix(mu.mean(),tc='d')
            w = solvers.qp(P,q,G, h, A, b)['x']
            w = np.array(w).T
            w = w/w.sum()
            return w



        self.backtest_ret = None
        # backtesting:       
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
            Omega_FC = self._factor_cov(df_train)
            
            # simple estimate covariance matrix:
            Omega_S = np.cov(df_train.T)
            mu = df_train.apply(np.mean,axis=0)
            
            # Minimum Variance Portfolio:
            w_FC[i,:] = tang_contrained(Omega_FC,mu)
            w_S[i,:] =  tang_contrained(Omega_S,mu)


        # backtesting 
        backtest_ret = self._get_ret_by_weight(w_FC,w_S)

        # add back rf

        self.backtest_ret = backtest_ret

        return backtest_ret

    
    
    def sample_sd(self):
        if self.backtest_ret is None:
            print('Run backtesting first')
            return None
        else:
            fac_std = np.std(self.backtest_ret.iloc[:,0])*np.sqrt(self.N)
            simply_std = np.std(self.backtest_ret.iloc[:,1])*np.sqrt(self.N)
            return {'freq':self.frequency, 'M':self.window, 'fac':self.factors,\
                    'FC':fac_std,\
                    'Simply':simply_std}

    def price(self):
        if self.backtest_ret is None:
            print('Run backtesting first')
            return None

        p = self.backtest_ret.join(self.rf,how='left')
        p['FC'] += p['RF']
        p['Simply'] += p['RF']
        p['Mkt'] = p['Mkt-RF'] + p['RF']
        p = p[['FC', 'Simply', 'Mkt']]
        return p.cumsum(axis = 0).apply(np.exp,axis=0)

         
    def vol_ewma(self,M,lambd):
        if self.backtest_ret is None:
            print('Run backtesting first')
            return None

        # calculate EWMA of portfolio returns
        # first update
        variance = np.zeros((self.backtest_ret.shape[0]-M,self.backtest_ret.shape[1]))
        variance_init = self.backtest_ret.iloc[:M,:].apply(np.var,axis=0)
        mu_init = self.backtest_ret.iloc[:M,:].apply(np.mean,axis=0)
        variance[0,:] = lambd * variance_init + (1-lambd)* (self.backtest_ret.iloc[M,]-mu_init)**2
        
        # 1-3 2-4 3-5   m=3 n=5
        for i in range(1,self.backtest_ret.shape[0]-M):
            mu = self.backtest_ret.iloc[i:M+i,:].apply(np.mean,axis=0)
            variance[i,:] = lambd * variance[i-1,:] + (1-lambd)* (self.backtest_ret.iloc[i+M,]-mu)**2
        
        # annulize
        sigma = np.sqrt(variance*self.N)
        sigma = pd.DataFrame(sigma,index=self.backtest_ret.iloc[M:,].index,columns =['FC','Simply','Mkt-RF'])

        return sigma
    
    def sharpe_ratio(self):
        if self.backtest_ret is None:
            print('Run backtesting first')
            return None

        mu = self.backtest_ret.apply(np.mean,axis=0)*self.N
        sd = self.backtest_ret.apply(np.std,axis=0)*np.sqrt(self.N)
        return mu/sd

    def alpha_beta(self):
        FC_regr = linear_model.LinearRegression().fit(self.backtest_ret[['Mkt-RF']],self.backtest_ret['FC'])
        FC_ab = {'Port':'FC','beta':FC_regr.coef_[0],'alpha': FC_regr.intercept_*self.N}
        S_regr = linear_model.LinearRegression().fit(self.backtest_ret[['Mkt-RF']],self.backtest_ret['Simply'])
        S_ab = {'Port':'S','beta':S_regr.coef_[0],'alpha': S_regr.intercept_*self.N}
        return [FC_ab,S_ab]

    def IR(self):
        def _IR(returns):     
            return_difference = returns - self.backtest_ret['Mkt-RF'] 
            volatility = return_difference.std() * np.sqrt(self.N) 
            information_ratio = return_difference.mean()*self.N / volatility
            return information_ratio
        return {'FC':_IR(self.backtest_ret['FC']), 'Simply':_IR(self.backtest_ret['Simply']) }

    def sortino_ratio(self):
        def lpm(returns, threshold, order):
            # This method returns a lower partial moment of the returns
            # Create an array he same length as returns containing the minimum return threshold
            threshold_array = np.empty(len(returns))
            threshold_array.fill(threshold)
            # Calculate the difference between the threshold and the returns
            diff = threshold_array - returns
            # Set the minimum of each to 0
            diff = diff.clip(lower=0)
            # Return the sum of the different to the power of order
            return np.sum(diff ** order) / len(returns)

        def sortino_ratio(er, returns, target=0):
            return (er) / math.sqrt(lpm(returns, target, 2))

        return {'FC': sortino_ratio(self.backtest_ret['FC'].mean(), self.backtest_ret['FC'])*np.sqrt(self.N),\
                'S': sortino_ratio(self.backtest_ret['Simply'].mean(), self.backtest_ret['Simply'])*np.sqrt(self.N)}


    
    