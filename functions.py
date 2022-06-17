"""

Copyright (C) 2021, Manel Vila-Vidal
Contact details: m@vila-vidal.com / manel.vila-vidal@upf.edu

Github: https://github.com/mvilavidal/localglobal2022

Created on Thu Apr 15 12:30:57 2021

-------------------------------------------------------------------

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License v3.0 as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License v3.0 for more details.

You should have received a copy of the GNU General Public License v2.0 along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

-------------------------------------------------------------------

ooo If you use the source code, please make sure to reference both the package and the paper:
    
> Vila-Vidal, M. (2021). Local-global 2022, https://github.com/mvilavidal/localglobal2022. Zenodo. (doi)

> Manel Vila-Vidal, Mariam Khawaja, Mar Carreño, Pedro Roldán, Jordi Rumià, Antonio Donaire, Gustavo Deco, Adrià Tauste Campo (2022). Assessing the coupling between local neural activity and global connectivity fluctuations: Application to human intracranial EEG during a cognitive task. bioRxiv, https://doi.org/10.1101/2021.06.25.449912.

"""

import numpy as np
import os, sys
import pickle
#import scipy.io
from scipy import signal, stats
import re
import pyedflib
from scipy.stats import pearsonr,spearmanr
import pandas as pd
from pingouin import partial_corr
import mne





#%%

###
def is_spiky(arr,consecutive=10):
    indices=[]
    inevent=False
    for i,val in enumerate(arr):
        if val==True:
            if inevent:
                newevent.append(i)
            else:
                newevent=[i]
                inevent=True
        else:
            if inevent:
                indices.append(newevent)
                inevent=False

    if len(indices) == 0:
        return False
    return any([len(idx) > consecutive for idx in indices])


def readEDF(filename,readdata=True,electrodes=None,timestart=0,timespan=None):
    with pyedflib.EdfReader(filename) as f:
        signalLabels=f.getSignalLabels()
        contacts=signalLabels
        contactelectrode=[re.split('(\d+)',c)[0] for c in contacts]
        if electrodes is None:
            electrodes=set(contactelectrode)
        idx=[i for i in range(len(contacts)) if contactelectrode[i] in electrodes]
        contacts=[contacts[i] for i in idx]
        fs=f.getSampleFrequencies()[0]
        Nc=len(contacts) # Number of contacts

        start=timestart*fs
        if timespan is None: 
            Nt=f.getNSamples()[0]-timestart
        else:
            Nt=timespan*fs
            
        if readdata:
            data=np.zeros((Nc,Nt))
        for i in range(Nc):
            c=contacts[i]
            j=signalLabels.index(c)
            if readdata:
                data[i]=f.readSignal(j,start=start,n=Nt) 
        j=signalLabels.index('TRIG')
        if readdata:
            trigger=f.readSignal(j,start=start,n=Nt,digital=True)
    
    contactelectrode=[re.split('(\d+)',c)[0] for c in contacts]
    Nel=len(set(contactelectrode))
    
    if readdata:
        return data,trigger,fs,Nt,Nc,contacts,Nel,contactelectrode
    else:
        return fs,Nt,Nc,contacts,Nel,contactelectrode

    
def refbipolar(data,contacts):
    Nc,Nt=data.shape
    contactelectrode=[re.split('(\d+)',c)[0] for c in contacts]
    Nel=len(set(contactelectrode))
    
    Nc2=Nc-Nel
    data2=np.zeros((Nc2,Nt))
    contacts2=[]
    contactelectrode2=[]
    i=0
    for j in range(Nc-1):
        el0=contactelectrode[j]
        el1=contactelectrode[j+1]
        if el0==el1:
            data2[i]=data[j]-data[j+1]
            contacts2+=[contacts[j]+'-'+contacts[j+1]]
            i+=1
            contactelectrode2+=[contactelectrode[j]]
    
    return data2,Nc2,contacts2,contactelectrode2


def cwm(contacts,where,bad):
    Nc=len(contacts)
    contactelectrode=[re.split('(\d+)',c)[0] for c in contacts]

    closestwm=np.ones(Nc,dtype=np.int)*-2
    for el in set(contactelectrode):
        idx=(np.array(contactelectrode)==el).nonzero()[0]
        Elength=idx.shape[0]
        for i in idx:
            c=contacts[i]
            if where[c].startswith('wm'): closestwm[i]=i
            elif where[c].startswith('out'): closestwm[i]=-1
            else:
                dist=np.abs(idx-i)
                for d in range(1,Elength):
                    idx2=idx[dist==d]
                    for j in idx2:
                        c=contacts[j]
                        if where[c]=='wm':
                            if c not in bad:
                                closestwm[i]=j
                    if closestwm[i]!=-2:
                        break
    return closestwm


def refwhitematter(data,bad,contacts,where):

    Nc,Nt=data.shape
    contactelectrode=[re.split('(\d+)',c)[0] for c in contacts]
    
    closestwm=cwm(contacts,where,bad)
    
    mask=((closestwm>-1)*(closestwm!=np.arange(Nc)))
    for c in bad:
        i=contacts.index(c)
        mask[i]=False
    gmidx=mask.nonzero()[0]
    Nc2=mask.sum()
    
    data2=np.zeros((Nc2,Nt))
    contacts2=[]
    contactelectrode2=[]
    
    for i in range(Nc2):
        j=gmidx[i]
        jref=closestwm[j]
        data2[i]=data[j]-data[jref]
        contacts2+=[contacts[j]]
        contactelectrode2+=[contactelectrode[j]]
    
    return data2,Nc2,contacts2,contactelectrode2



def keepgm(data,bad,contacts,where):


    Nc,Nt=data.shape
    contactelectrode=[re.split('(\d+)',c)[0] for c in contacts]
    
    gray=np.zeros(Nc,dtype=np.bool)
    for k in range(Nc):
        c=contacts[k]
        if where[c].startswith('wm'):
            continue
        elif where[c].startswith('out'):
            continue
        elif where[c].endswith('?'):
            continue
        else:
            if c not in bad:
                gray[k]=True

    grayidx=gray.nonzero()[0]
    Nc2=gray.sum()
    
    data2=np.zeros((Nc2,Nt))
    contacts2=[]
    contactelectrode2=[]
    
    for i in range(Nc2):
        j=grayidx[i]
        data2[i]=data[j]
        contacts2+=[contacts[j]]
        contactelectrode2+=[contactelectrode[j]]
    
    return data2,Nc2,contacts2,contactelectrode2   




def subfinder(mylist, pattern):
    matches = []
    L=len(pattern)
    for i in range(len(mylist)-L+1):
        if all(mylist[i:i+L] == pattern):
            matches.append(i)
    return matches


#%% Filters


def _data_filter(data,fs,l_freq,h_freq):
    y=data
    y=(y.T-y.mean(axis=-1)).T # demean
    y=signal.detrend(y,axis=-1) # detrend
    y=mne.filter.notch_filter(y,fs,np.arange(50, fs/2, 50),n_jobs=6)
    y=mne.filter.filter_data(y,fs,l_freq,h_freq,n_jobs=6)
    return y


#%%



def _joblib_wrapper_cplv_singletrial(data_angle,k1,k2,window_len,window_step):
    phase_diff = np.exp(1j*(data_angle[:,k1] - data_angle[:,k2]))
    cplv=np.array([np.mean(phase_diff[:,wi*window_step:wi*window_step+window_len],axis=1) for wi in range(Nw)])
    return cplv.T



def fisherZ(r):
    return 0.5*np.log( (1+r)/(1-r) )

def fisherZinv(z):
    return (np.exp(2*z)-1)/(np.exp(2*z)+1)


#%%


eps=sys.float_info.epsilon


def spearmanr_circular_shifts(x,y,lag_min,lag_max=None,N_shuffles=1000):
    x=np.array(x)
    y=np.array(y)
    N=x.shape[-1]
    if lag_max is None: lag_max=N-lag_min
    lags=np.random.randint(lag_min,lag_max,N_shuffles)
    dist=np.zeros(N_shuffles)
    for i,lag in enumerate(lags):
        y_surr=np.roll(y,lag)
        (r,p)=spearmanr(x,y_surr)
        dist[i]=r
    (r,p)=spearmanr(x,y)
    p=(np.abs(r)<=np.abs(dist)).sum()/N_shuffles
    return (r,p)





def spearmanr_partial_circular_shifts(locs,localind,glob,lag_min,lag_max=None,N_shuffles=1000):
    
    (Nc,N)=locs.shape
    covs=[str(i) for i in range(Nc) if i!=localind]
    data=pd.DataFrame(np.concatenate((locs.T,glob[:,np.newaxis]),axis=1),columns=[str(i) for i in range(Nc)]+['global'])
    #res=partial_corr(data,x=str(localind),y='global',y_covar=covs,method='spearman')
    res=partial_corr(data,x=str(localind),y='global',covar=covs,method='spearman')
    r=res.r.values[0]
    
    if lag_max is None: lag_max=N-lag_min
    lags=np.random.randint(lag_min,lag_max,N_shuffles)
    dist=np.zeros(N_shuffles)
    for i,lag in enumerate(lags):
        glob_surr=np.roll(glob,lag)
        data_surr=pd.DataFrame(np.concatenate((locs.T,glob_surr[:,np.newaxis]),axis=1),columns=[str(i) for i in range(Nc)]+['global'])
        #res_surr=partial_corr(data_surr,x=str(localind),y='global',y_covar=covs,method='spearman')
        res_surr=partial_corr(data_surr,x=str(localind),y='global',covar=covs,method='spearman')
        r_surr=res_surr.r.values[0]
        dist[i]=r_surr
    p=(np.abs(r)<=np.abs(dist)).sum()/N_shuffles
    return (r,p)

