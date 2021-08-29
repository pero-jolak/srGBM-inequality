# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:50:20 2021

@author: Petar Jolakoski
"""

#%%
import os
os.chdir('C:/Users/W10/Desktop/srgbm')
#%% import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import gmean
import statsmodels.api as sm
from scipy.optimize import minimize
import scipy.stats as st

#%% read data

ddp_ = pd.read_excel('perc_data.xlsx', sheet_name='Sheet2')
perc_data = pd.read_excel('perc_data.xlsx',sheet_name='Sheet1')

ddp = np.array(ddp_[ddp_['year'].isin(range(1977,2016,1))].ddp)
rs_ = pd.read_excel('r.xlsx')
rs = np.array(rs_.r1)
rs2 = np.array(rs_.r2)
rs3 = np.array(rs_.r3)

data_share = pd.read_excel('data_share.xlsx')
data_sh = np.array(data_share[data_share['year'].isin(range(1977,2016,1))]['share'])
#%% read data share for top 10% earners

data_share_ = pd.read_excel('data_share_10.xlsx')
data_sh10 = np.array(data_share_[data_share_['year'].isin(range(1977,2016,1))]['share'])
#%% read data share for top 0.1% earners

data_share_001 = pd.read_excel('data_share001.xlsx')
data_sh001 = np.array(data_share_001.share001)

#%% check len(data)

print(len(ddp))
print(len(rs))
print(len(rs2))
print(len(data_sh))
print(len(data_sh10))
#%% import percentile data for 1977 (t=0) in order to generate the initial N trajectories from the empirical income distribution

data = pd.read_excel('perc_data.xlsx')
data__ = data[data['year']==1977].avg_income
hist = np.histogram(data__)
hist_dist = st.rv_histogram(hist)

x = hist_dist.rvs(size=50000)
data_ = x
data_.sort()

print(np.exp(np.mean(np.log(data_))))
#%% create a function for MLE estimation

def neg_log_lik(theta):
  lik = []

  r = 0.00626144443706745
  mu = theta[0]
  ssq = 0.16**2
  
  alpha = (-(mu-ssq/2)+np.sqrt((mu-ssq/2)**2+2*r*ssq))/ssq

  x0 = np.exp(np.mean(np.log(data_)))
  
  for i in range(len(data_)):
      if data_[i]>x0:
          lik.append(((r*ssq)/(alpha*ssq+(mu-ssq/2))) * (data_[i]/x0)**(-alpha-1))
      else:
          lik.append(((r*ssq)/(alpha*ssq+(mu-ssq/2))) * (data_[i]/x0)**(alpha+2*(mu-ssq/2)-1))
  
  return -sum(np.log(lik))
#%%

theta_start = np.array([0.015])
res = minimize(neg_log_lik, theta_start, method = 'Nelder-Mead', 
	       options={'disp': True,'maxiter':100000})

print(res)          
#%% Estimate the standard deviation in each year (you can use this as an estimate for sigma) -- in the paper it is not used

from sklearn import preprocessing
minmax = preprocessing.MinMaxScaler()
stds = []

for t in range(1977,2016,1):
    data = pd.read_excel('perc_data.xlsx')
    data_ = data[data['year']==t].avg_income
    #hist = np.histogram(data_)
    #hist_dist = st.rv_histogram(hist)

    #x = hist_dist.rvs(size=10000)
    #data_ = x
    
    data__ = np.array(data_).reshape(len(data_),1)
    
    stds.append(np.std(minmax.fit_transform(data__)))

#%% Fit mu in n iterations (take avg and then std for error bars)

iterations = 1

fitted_mu = np.ones([38, iterations])
fitted_ddps = np.ones([38, iterations])

for iteration in range(iterations):
    print("Iteration: "f"{iteration}")
    
    min_location = []
    min_pred = []
    min_sh = []
    
    random.seed(10)
    t = len(ddp)
    people = 50000
    dt = 1
    
    sigma = np.sqrt(0.02219277)
    trajs = np.ones([100, t+1, people])
    
    for real in range(0,t-1,1):
      print(real)
    
      sh = []
      sqerror = []
      pred = []
    
      if real == 0:
        trajs[:,0,:] = data_
        mus = np.linspace(0.01974285,0.01974285,100)
      else:
        mus = np.linspace(0.001,0.10,100)
        
      for ri, mu in enumerate(mus):
        
        choice_ = [[0,1],[1-rs[real+1]*dt, rs[real+1]*dt]]
        prob = np.random.choice(a=choice_[0], p=choice_[1], size=people)
        noise = np.random.randn(1,people)
    
        trajs[ri, real + 1, np.argwhere(prob == 0)] = trajs[ri, real, np.argwhere(prob == 0)] * (1 + mu * dt + (sigmas[real+1] * np.sqrt(dt)) * noise[0,np.argwhere(prob == 0)])
        trajs[ri, real + 1, np.argwhere(prob == 1)] = trajs[ri, 0, np.argwhere(prob == 1)]
        
        check = trajs[ri, real + 1, :]
        trajs[ri, real + 1, np.where(check<0)] = 1 #trajs[ri, 0, np.where(check<0)]
    
        trajs[ri, real+1, :].sort()
        
        #share = np.sum(trajs[ri, real+1,:][-100:])/np.sum(trajs[ri, real+1,:])
        share = np.sum(trajs[ri, real+1,:][-1000:])/np.sum(trajs[ri, real+1,:])
        #ddps = np.exp(np.mean(np.log(trajs[ri,real+1,:])))
    
        #sqerror.append((share-data_sh[real+1])**2)
        sqerror.append((share-data_sh10[real+1])**2)
        #sqerror.append((ddps-ddp[real+1])**2)
        
        sh.append(share)
        #pred.append(ddps)
    
      min_loc = sqerror.index(min(sqerror))
      #min_pred.append(pred[min_loc])
      min_location.append(mus[min_loc])
      min_sh.append(sh[min_loc])
     
    fitted_mu[:, iteration] = min_location
    fitted_ddps[:, iteration] = min_sh


#%% Fit mu and sigma together in n iterations (take avg and then std for error bars)

iterations = 50

fitted_mu = np.ones([38, iterations])
fitted_sigma = np.ones([38, iterations])
fitted_ddps = np.ones([38, iterations])

for iteration in range(iterations):
    print("Iteration: "f"{iteration}")   
    min_location = []
    min_pred = []
    min_sh = []
    
    random.seed(10)
    t = len(data_sh001)
    people = 50000
    dt = 1
    m = 50
    #sigma = np.sqrt(0.02219277)
    #sigma = 0.2
    trajs = np.ones([m, m, t+1, people],dtype='uint8')
    
    for real in range(0,t-1,1):
      print(real)
    
      sh = []
      sqerror = []
      pred = []
    
      if real == 0:
        trajs[:,0,:] = data_
        mus = np.linspace(0.01974285,0.01974285,m)
        sigmas = np.linspace(np.sqrt(0.02219277),np.sqrt(0.02219277),m)
      else:   
        mus = np.linspace(0.001,0.15,m)
        sigmas = np.linspace(0.10,0.70,m)
      lista = {}
      
      for muiter, mu in enumerate(mus):
    
          for siter, sigma in enumerate(sigmas):
              
              choice_ = [[0,1],[1-rs3[real+1]*dt, rs3[real+1]*dt]]
              prob = np.random.choice(a=choice_[0], p=choice_[1], size=people)
              noise = np.random.randn(1,people)
    
              trajs[muiter,siter, real + 1, np.argwhere(prob == 0)] = trajs[muiter,siter, real, np.argwhere(prob == 0)] * (1 + mu * dt + (sigma * np.sqrt(dt)) * noise[0,np.argwhere(prob == 0)])
              trajs[muiter,siter, real + 1, np.argwhere(prob == 1)] = trajs[muiter,siter, 0, np.argwhere(prob == 1)]
        
              check = trajs[muiter,siter, real + 1, :]
              trajs[muiter,siter, real + 1, np.where(check<0)] = 1 #trajs[ri, 0, np.where(check<0)]
    
              trajs[muiter,siter, real+1, :].sort()
        
              share = np.sum(trajs[muiter, siter, real+1,:][-50:])/np.sum(trajs[muiter, siter, real+1,:])
              #ddps = np.exp(np.mean(np.log(trajs[muiter,siter,real+1,:])))
    
              #sqerror.append((ddps-ddp[real+1])**2)
              sqerror.append((share-data_sh001[real+1])**2)
    
              sh.append(share)
              #pred.append(ddps)    
              
              lista[muiter,siter] = mu, sigma             
      min_loc = sqerror.index(min(sqerror))
      #min_pred.append(pred[min_loc])
      min_location.append(list(lista.values())[min_loc])
      min_sh.append(sh[min_loc])
    
    fitted_mu[:, iteration] = np.array(pd.DataFrame(min_location)[0])
    fitted_sigma[:, iteration] = np.array(pd.DataFrame(min_location)[1])
    fitted_ddps[:, iteration] = min_sh
    
#%%
pd.DataFrame(np.mean(fitted_mu, axis=1)).to_excel('results/top10/r1/fitted_mu.xlsx')
pd.DataFrame(np.mean(fitted_sigma, axis=1)).to_excel('results/top10/r1/fitted_sigma.xlsx')
pd.DataFrame(np.std(fitted_mu, axis=1)).to_excel('results/top10/r1/mu_together_stderror.xlsx')
pd.DataFrame(np.std(fitted_sigma, axis=1)).to_excel('results/top10/r1/sigma_together_stderror.xlsx')

pd.DataFrame(np.mean(fitted_ddps, axis=1)).to_excel('results/top10/r1/fitted_top1.xlsx')
pd.DataFrame(np.std(fitted_ddps, axis=1)).to_excel('results/top10/r1/fitted_top1_stderror.xlsx')
#%%
pd.DataFrame(np.mean(fitted_mu, axis=1)).to_excel('results/top10/r2/fitted_mu.xlsx')
pd.DataFrame(np.mean(fitted_sigma, axis=1)).to_excel('results/top10/r2/fitted_sigma.xlsx')
pd.DataFrame(np.std(fitted_mu, axis=1)).to_excel('results/top10/r2/mu_together_stderror.xlsx')
pd.DataFrame(np.std(fitted_sigma, axis=1)).to_excel('results/top10/r2/sigma_together_stderror.xlsx')

pd.DataFrame(np.mean(fitted_ddps, axis=1)).to_excel('results/top10/r2/fitted_top1.xlsx')
pd.DataFrame(np.std(fitted_ddps, axis=1)).to_excel('results/top10/r2/fitted_top1_stderror.xlsx')
#%%
pd.DataFrame(np.mean(fitted_mu, axis=1)).to_excel('results/top001/r3/fitted_mu.xlsx')
pd.DataFrame(np.mean(fitted_sigma, axis=1)).to_excel('results/top001/r3/fitted_sigma.xlsx')
pd.DataFrame(np.std(fitted_mu, axis=1)).to_excel('results/top001/r3/mu_together_stderror.xlsx')
pd.DataFrame(np.std(fitted_sigma, axis=1)).to_excel('results/top001/r3/sigma_together_stderror.xlsx')

pd.DataFrame(np.mean(fitted_ddps, axis=1)).to_excel('results/top001/r3/fitted_top1.xlsx')
pd.DataFrame(np.std(fitted_ddps, axis=1)).to_excel('results/top001/r3/fitted_top1_stderror.xlsx')
#%%

#%% srGBM with numerical simulations


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.integrate import cumtrapz
import scipy.optimize as opt
from scipy.stats.mstats import gmean


sigma = np.sqrt(0.01)
mu = 0.02

t = 1000
nt = t
dt = 1
ntraj = 10000

trajs = np.ones([nt, ntraj])
ddp = []
sh = []

r = 0.01

for i in range(nt):
    if i < nt - 1:
      choice_ = [[0,1],[1-r*dt, r*dt]]
      prob = np.random.choice(a=choice_[0], p=choice_[1], size=ntraj)
      noise = np.random.randn(1,ntraj)

      trajs[i + 1, np.argwhere(prob == 0)] = trajs[i,np.argwhere(prob == 0)] * (1 + mu * dt + (sigma * np.sqrt(dt)) * noise[0,np.argwhere(prob == 0)])
      trajs[i + 1, np.argwhere(prob == 1)] = trajs[0,np.argwhere(prob == 1)]

      ddp.append(np.exp(np.mean(np.log(trajs[i+1,:]))))
      trajs[i+1,:].sort()
      sh.append(np.sum(trajs[i+1,:][-100:])/np.sum(trajs[i+1,:]))

      print(i)
#%% Checking the fitting method

sigma = np.sqrt(0.01)
#mu = 0.02

t = 1000
nt = t
dt = 1
ntraj = 10000

trajs = np.ones([50, nt, ntraj])#,dtype='uint8')
#ddp = []
#sh = []
min_location = []
min_pred = []
r = 0.01
mus = np.linspace(0.001,0.05,50)

for i in range(nt-1):
    if i < nt - 1:
        
      sqerror = []
      pred = []
      
      for ri, mu in enumerate(mus):
          
        choice_ = [[0,1],[1-r*dt, r*dt]]
        prob = np.random.choice(a=choice_[0], p=choice_[1], size=ntraj)
        noise = np.random.randn(1,ntraj)

        trajs[ri, i + 1, np.argwhere(prob == 0)] = trajs[ri, i,np.argwhere(prob == 0)] * (1 + mu * dt + (sigma * np.sqrt(dt)) * noise[0,np.argwhere(prob == 0)])
        trajs[ri, i + 1, np.argwhere(prob == 1)] = trajs[ri, 0,np.argwhere(prob == 1)]
        
        check = trajs[ri, i + 1, :]
        trajs[ri, i + 1, np.where(check<0)] = 1
        
        trajs[ri, i+1,:].sort()

        share = np.sum(trajs[ri, i+1,:][-100:])/np.sum(trajs[ri, i+1,:])
        #ddps = np.exp(np.mean(np.log(trajs[ri, i+1,:])))
    
        sqerror.append((share-sh[i+1])**2)
        #sqerror.append((ddps-ddp[i+1])**2)
        
        pred.append(share)
        
      min_loc = sqerror.index(min(sqerror))
      min_pred.append(pred[min_loc])
      min_location.append(mus[min_loc])
      print(i)
#%%

plt.plot(min_pred)
plt.plot(sh)