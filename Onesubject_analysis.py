#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

#%% Execute first



### Imports and global vars

#%load_ext autoreload
#%autoreload 2
from IPython.core.debugger import set_trace
import pyedflib
import matplotlib.pyplot as plt
import os, sys
import scipy.io
import numpy as np
from scipy import signal, stats
import re
from importlib import reload
from scipy.signal import savgol_filter
import mplcursors
import sys
import pandas as pd
#%matplotlib notebook
import mne
import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm
from scipy.stats import pearsonr,spearmanr
import scipy.stats as st

# Get parent directory where functions are
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from functions import *


# Get data directory
mydir=os.environ['HOME']+'/wd'
datadir=mydir+'/Data'


pat='P5-BCN'
date='13_10_2020'
session='Screening1_test'
# Get/create output directory (not updated in git)
outputdir=f'{mydir}/BIAL github/{pat}-{date}/output'
if not os.path.isdir(outputdir): os.mkdir(outputdir)


bad=['FA12','FM2']
verbose=1





testevents=[1,2,5,8,17,32,65,128,85,84,4,16,0]

paradigmonset=[85, 84, 85, 84, 85, 84]
picture1=[[3, 0],
            [9, 0],
            [33, 0]]
picturenext=[[1, 0],
            [5, 0],
            [17, 0]]

picture=picture1+picturenext

linesoff=[77, 0]
joystick=[81, 0]

timeover=[83, 0]
sequenceend=[113,0]

linechange=[[103,0],
            [133,0]]



### Load data

bipolar=0
wmref=0
comrefgm=1

# Experiment data ### N/A
# filepath=f'{datadir}/BIAL/{pat}-{date}/{pat}-{session}_{date}/000rsvpscr/order_pics_RSVP_SCR.mat'
# mat=scipy.io.loadmat(filepath,squeeze_me=False)#,chars_as_strings=None)
# Nrep=mat['Nrep'][0,0]
# ISI=mat['ISI'][0,0]
# order_pics=mat['order_pic']
# shape=order_pics.shape
# seq_length=shape[0]
# Nseq=shape[-1] if len(shape)==3 else 1
# Npics=len(set(order_pics.flatten()))
# pic_names=scipy.io.loadmat(f'{datadir}/BIAL/{pat}-{date}/{pat}-{session}_{date}/000rsvpscr/outputs/RSVP_SCR_workspace.mat',squeeze_me=True)['ImageNames']
# times=scipy.io.loadmat(f'{datadir}/BIAL/{pat}-{date}/{pat}-{session}_{date}/000rsvpscr/outputs/timesanswer-22-Sep-2020-16-38-44.mat',squeeze_me=True)['times']

# Contact locations
filepath=f'{datadir}/BIAL/{pat}-{date}/{pat}_contact_correspondence_mvv.xlsx'
sheet='TDT'
df = pd.read_excel(io=filepath, sheet_name=sheet, na_values=['[]'],usecols='A,B,D',names=['Label','DK','MyAtlas'])
whereDK={}
whereMVV={}
for i in range(df.shape[0]):
    whereDK[df.Label[i]]=df.DK[i]
    whereMVV[df.Label[i]]=df.MyAtlas[i]
    
# Read recordings from EDF file
filepath=f'{datadir}/BIAL/{pat}-{date}/{pat}-{session}_{date}.EDF'
data,trigger,fs,Nt,Nc,contacts,Nel,contactelectrode=readEDF(filepath,readdata=True,electrodes=['FB', 'FA', 'FM', 'FP', 'WR', 'AMS', 'Ti'],timespan=5*60)
if bipolar:
    data,Nc,contacts,contactelectrode=refbipolar(data,contacts)
elif wmref:
    data,Nc,contacts,contactelectrode=refwhitematter(data,bad,contacts,whereDK)
elif comrefgm:
    data,Nc,contacts,contactelectrode=keepgm(data,bad,contacts,whereDK)
print(data.shape,set(contactelectrode))


# ###### Demean, detrend, notch
# y=data
# y=(y.T-y.mean(axis=-1)).T # demean
# y=signal.detrend(y,axis=-1) # detrend
# sos = signal.butter(2, [0.1,1000], btype='bandpass',fs=fs, output='sos') # remove slow drifts
# y = signal.sosfiltfilt(sos, y, axis=-1)
# ynotch=notchfilter(y,fs,notchauto=True) # notch filter

# ###### Plot periodogram
# plt.figure('PSD one channel')
# n=2*fs
# f,Pxx_den=signal.welch(y[1],nperseg=n,fs=fs,nfft=n)
# a=10*np.log10(Pxx_den)
# plt.plot(f,a)
# f,Pxx_den=signal.welch(ynotch[1],nperseg=n,fs=fs,nfft=n)
# a=10*np.log10(Pxx_den)
# plt.plot(f,a)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.xlim([0,800])
# plt.ylim([-30,30])
# #plt.xlim([0,100])
# #plt.ylim([-15,15])
# plt.show()

# y=ynotch


# Contact pairs

contactpairs=[]
for k1 ,c1 in enumerate(contacts):
    for k2 ,c2 in enumerate(contacts):
        if contactelectrode[k1]!=contactelectrode[k2]:
            if (k2,k1) not in contactpairs:
                contactpairs.append((k1,k2))
        else:
            if whereDK[c1]!=whereDK[c2]:
                if (k2,k1) not in contactpairs:
                    contactpairs.append((k1,k2))
K1s,K2s=zip(*contactpairs)


### Events from trigger

e=trigger[0]
i=0
events=[e]
eventstidx=[i]
T=trigger.shape[0]
assert Nt==T
for i in range(T):
    if trigger[i]!=e:
        e=trigger[i]
        events.append(e)
        eventstidx.append(i)
events=np.array(events)
eventstidx=np.array(eventstidx)

if verbose:
    print('Events:\n',events,end='\n\n')
    plt.figure('trigger')
    plt.plot(trigger)
    plt.plot(eventstidx,events,'r+')

# Extract events for paradigm duration without line changes

# Find paradigm onset
ionset=subfinder(events,paradigmonset)
ionset=ionset[0]
# Find end of each sequence and paradigm offset
iendseqs=subfinder(events,sequenceend)
#!!!!!! assert len(iendseqs)==Nseq
ioffset=iendseqs[-1]

auxevents=[]
auxeventstidx=[]
skip=0
for i in range(ionset,ioffset+2):
    if list(events[i:i+2]) in linechange:
        skip=1
    elif skip>0:
        skip-=1
    else:
        auxevents+=[events[i]]
        auxeventstidx+=[eventstidx[i]]
events=np.array(auxevents)
eventstidx=np.array(auxeventstidx)


if verbose:
    print('Events:\n',events,end='\n\n')
    plt.figure('trigger without line changes')
    plt.plot(trigger)
    plt.plot(eventstidx,events,'r+')



# Time indices images

trial=0
picstidx=[]
picseventsidx=[]
endtrialtidx=[]
aux=[]

sign1=joystick
sign0=timeover
trials1=[]
trials0=[]

deltapic=[] # time that the picture remains (in ms)
deltajoystick=[] # for Recognized: time between the picture appears and the joystick is pressed (in ms)
deltajoysticknext=[] # for Recognized: time between tthe joystick is pressed and the next picture appears (in ms)
deltatimeover=[] # for non Recognied: time between the picture apperars and time is over (in ms)
deltatimeovernext=[] # for non Recognized: time between time over and the next picture appears (in ms)

previous=-1

for i in range(len(events)-1):
    
    if list(events[i:i+2]) in picture:
        picstidx.append(eventstidx[i])
        picseventsidx.append(i)
        aux.append(i)
        tpicon=eventstidx[i]/fs
        if previous==1:
            deltajoysticknext.append(tpicon-tprevioustrialend)
        elif previous==0:
            deltatimeovernext.append(tpicon-tprevioustrialend)

        
        for j in range(i,len(events)-1):
            
            if list(events[j:j+2])==linesoff:
                tpicoff=eventstidx[j]/fs
                deltapic.append(tpicoff-tpicon)
            
            if list(events[j:j+2])==sign1:
                trials1.append(trial)
                endtrialtidx.append(eventstidx[j])
                trial+=1
                t1=eventstidx[j]/fs
                deltajoystick.append(t1-tpicoff)
                previous=1
                tprevioustrialend=eventstidx[j]/fs
                break
            
            if list(events[j:j+2])==sign0:
                trials0.append(trial)
                endtrialtidx.append(eventstidx[j])
                trial+=1
                t0=eventstidx[j]/fs
                deltatimeover.append(t0-tpicoff)
                previous=0
                tprevioustrialend=eventstidx[j]/fs
                break            

picstidx=np.array(picstidx)
endtrialtidx=np.array(endtrialtidx)
Npics=len(picstidx)
N1=len(trials1)
N0=len(trials0)
assert N0+N1==Npics
assert Npics==len(deltapic)
assert N1==len(deltajoystick)
assert N0==len(deltatimeover)
assert Npics-1==len(deltatimeovernext)+len(deltajoysticknext)

print('Npics,N1,N0:',Npics,N1,N0)

deltapic=np.array(deltapic)
deltajoystick=np.array(deltajoystick)
deltatimeover=np.array(deltatimeover)
deltajoysticknext=np.array(deltajoysticknext)
deltatimeovernext=np.array(deltatimeovernext)

ISI=deltapic.mean()
print('ISI:',deltapic.mean(),deltapic.std())

# Pre-pics
Pre1=deltajoysticknext.mean() # time between joystick is pressed and next image appears
print('PRE1:',deltajoysticknext.mean(),deltajoysticknext.std())
deltatimeovernext.mean() # time between time over and next image appears
print('PRE0:',deltatimeovernext.mean(),deltatimeovernext.std()) 
Pre=np.concatenate((deltajoysticknext,deltajoysticknext)).mean()
print('PRE:',Pre) 
Pre=0.5
print('PRE:',Pre) 

# REC time before reply
TimeRFastest=deltajoystick.min()
TimeRSlowest=deltajoystick.max()
print('Time before reply (fastest,slowest):',deltajoystick.min(),deltajoystick.max())
# UNREC time left to reply
print('Timeout when not recognized:',deltatimeover.mean(),deltatimeover.std()) 
TimeOuT=deltatimeover.mean()


if verbose:
    plt.figure('trigger with marks when pics appear')
    plt.plot(trigger)
    plt.vlines(picstidx[trials1],plt.ylim()[0],plt.ylim()[1],'green')
    plt.hlines(sign1,plt.xlim()[0],plt.xlim()[1],'k')
    plt.vlines(picstidx[trials0],plt.ylim()[0],plt.ylim()[1],'red')
    plt.hlines(sign0,plt.xlim()[0],plt.xlim()[1],'k')
    plt.hlines([40],plt.xlim()[0],plt.xlim()[1],'k')
    plt.show()

Pre=0.5
Post=1.5

#%% BAD SAMPLES

#%%% Envelope wavelet

# Data
y=data
y=(y.T-y.mean(axis=-1)).T # demean
y=signal.detrend(y,axis=-1) # detrend
y=mne.filter.notch_filter(y,fs,np.arange(50, fs/2, 50),n_jobs=6)
l_freq=1
h_freq=700
y=mne.filter.filter_data(y,fs,l_freq,h_freq,n_jobs=8)

def get_frequencies():
    return 2**np.arange(1,np.log2(512+1),1/4)

# Wavelet
n_cycles=7
omega=n_cycles
frequencies = get_frequencies()
Nf=frequencies.shape[0]

data_envelope=np.zeros((Nf,Nc,Nt))
for fi, freq in enumerate(tqdm.tqdm(frequencies, leave=False, desc='Frequencies')):
    data_preprocessed=mne.time_frequency.tfr_array_morlet(y[np.newaxis,...], int(fs), [freq], omega, n_jobs=6).squeeze()
    data_envelope[fi] = np.abs(data_preprocessed)


#%%% Find bad samples

res_fname = os.path.join(outputdir, f'{pat}-{date}-{session}_bad_times.pickle')
if os.path.exists(res_fname):
    print('Already processed!')
else:
    # Params
    std_ratio=5
    window_size = int(fs/4)
    affected_frequencies_ratio = 0.5
    affected_channels_ratio = 0.1
    
    suspicious_frequencywise = np.zeros((Nf, Nc, Nt), dtype=bool)
    
    for fi, freq in enumerate(tqdm.tqdm(frequencies)):
        env = data_envelope[fi]
        threshold = env.mean(axis=-1, keepdims=True) + std_ratio*np.std(env, axis=-1, keepdims=True)
        
        suspicious_frequencywise[fi] = env >= threshold
        
        for i in np.arange(0, Nt, window_size):
            for k in range(Nc):
                suspicious_frequencywise[fi, k, i:i+window_size] = is_spiky(suspicious_frequencywise[fi, k, i:i+window_size],consecutive=10)
                      
    
    frequency_mean = suspicious_frequencywise.mean(axis=0)
    channel_mean = (frequency_mean >= affected_frequencies_ratio).mean(axis=0)
    bad_samples_mask = channel_mean >= affected_channels_ratio
    
    res_data = {'bad_samples': suspicious_frequencywise, 'mask': bad_samples_mask}
    pickle.dump(res_data, open(res_fname, 'wb'))
       
#%%%

res_fname = os.path.join(outputdir, f'{pat}-{date}-{session}_bad_times.pickle')
res_data = pickle.load(open(res_fname, 'rb'))
print(res_data['mask'].shape[0]/fs/60)
print(res_data['mask'].sum()/Nt)




#%%% Affected trials

# time points : PRE (500 ms), ISI (800 ms aprox), POST (we use Pre, 500ms)
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
 
affected_trials=0
for i in range(Npics):
     j=picstidx[i]
     affected_trials+=np.any(bad_samples_mask[j-Npre:j+Npost])
  
print(affected_trials)

#%% ERPs

#%%% ERP for report

c='AMS1'
#k=contacts.index('FP11')
#k=contacts.index('FP12')
#k=contacts.index('FP13')
k=contacts.index(c)


# Data
y=data[k]
y=(y.T-y.mean(axis=-1)).T # demean
y=signal.detrend(y,axis=-1) # detrend
y=mne.filter.notch_filter(y,fs,np.arange(50, fs/2, 50),n_jobs=8)
l_freq=1
h_freq=30 # or 80 the result is the same
y=mne.filter.filter_data(y,fs,l_freq,h_freq,n_jobs=6)


### ERps aligned to image onset
Nt_trials=int((Pre+ISI+Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T=np.arange(-Npre,Npost)/fs*1000


M=0
#k=7
plt.figure(0)
hlines=[]
ERP=np.zeros((Npics,Nt_trials))
for i in range(Npics):
     j=picstidx[i]
     erp=y[j-Npre:j+Npost]
     baseline=erp[:int(fs*0.5)].mean(axis=-1)
     #s=erp[:int(fs*0.5)].std(axis=-1)
     #erp=(erp-baseline)
     
     ERP[i]=erp
     
     
     
     # plt.plot(T,aux,color='k')
     # hlines.append(aux.mean()-50)
     # hlines.append(aux.mean()+50)

     # M=aux.max()
     
m=np.median(ERP,axis=0)
s=ERP.std(axis=0)/np.sqrt(Npics)
     

plt.figure(figsize=(10,6)) 
plt.fill_between(T, m-s, m+s, color='k', alpha=0.2)
plt.plot(T,m,'k')
plt.hlines(hlines,T[0],T[-1],color='k',linestyle='dashed')        
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
plt.title(contacts[k])
#plt.yticks(hlines,[-50,50]*Npics+[-20,20])
plt.xlabel('')
plt.ylabel('')
plt.title('')
xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(outputdir+'/ERP_'+c+'.png')


#%%% ERP for report aligned to response

c='FP13'
#k=contacts.index('FP12')
#k=contacts.index('FP13')
#k=contacts.index('AMS1')
k=contacts.index(c)

# Data
y=data[k]
y=(y.T-y.mean(axis=-1)).T # demean
y=signal.detrend(y,axis=-1) # detrend
y=mne.filter.notch_filter(y,fs,np.arange(50, fs/2, 50),n_jobs=8)
l_freq=1
h_freq=30 # or 80 the result is the same
y=mne.filter.filter_data(y,fs,l_freq,h_freq,n_jobs=6)


### ERps aligned to action
Nt_trials_action=int((2*Pre+Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000



M=0
#k=7
plt.figure(0)
hlines=[]
ERP_action=np.zeros((N1,Nt_trials_action))
for i in range(N1):
    trial=trials1[i]
    jpic=int(picstidx[trial])
    joystick=int(endtrialtidx[trial])   
    ##     
    erp=y[jpic-Npre:joystick+Npre]
    baseline=erp[:int(fs*0.5)].mean(axis=-1)
    #baseline=erp[-fs:-int(fs/2)].mean(axis=-1)
    #s=erp[:int(fs*0.5)].std(axis=-1)
    erp=(erp-baseline)
    
    ERP_action[i]=np.concatenate((erp[:int(fs/2)],erp[-fs:]))

ERP_noaction=np.zeros((N0,Nt_trials_action))
for i in range(N0):
    trial=trials0[i]
    jpic=int(picstidx[trial])
    timeout=int(endtrialtidx[trial])   
    ##     
    erp=y[jpic-Npre:timeout+Npre]
    #baseline=erp[:int(fs*0.5)].mean(axis=-1)
    #baseline=erp[-fs:-int(fs/2)].mean(axis=-1)
    #s=erp[:int(fs*0.5)].std(axis=-1)
    #erp=(erp-baseline)
    
    ERP_noaction[i]=np.concatenate((erp[:int(fs/2)],erp[-fs:]))
     
plt.figure(figsize=(10,6)) 

m=np.median(ERP_action,axis=0)
s=ERP_action.std(axis=0)/np.sqrt(N1)
plt.fill_between(T_action, m-s, m+s, color='b', alpha=0.2)
plt.plot(T_action,m,'b',label='Recognized')

m=np.median(ERP_noaction,axis=0)
s=ERP_noaction.std(axis=0)/np.sqrt(N0)
plt.fill_between(T_action, m-s, m+s, color='r', alpha=0.2)
plt.plot(T_action,m,'r',label='Unrecognized')
 
#plt.hlines(hlines,T[0],T[-1],color='k',linestyle='dashed')        
plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
#plt.vlines([-500],plt.ylim()[0],plt.ylim()[1],color='blue',linewidth=3)
plt.title(contacts[k])
#plt.yticks(hlines,[-50,50]*Npics+[-20,20])
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.legend(fontsize=18)
xlabels=['' for i in np.arange(-1000,501,100)]
#xlabels[1]='-400'
#xlabels[3]='-200'
#xlabels[5]='stimulus'
xlabels[6]='-400'
xlabels[8]='-200'
xlabels[10]='0'
xlabels[12]='200'
xlabels[14]='400'
plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(outputdir+'/ERP_'+c+'v3.png')



significance=np.zeros(int(1.5*fs))
for i in range(significance.shape[0]):
    significance[i]=st.ranksums(ERP_action[:,i],ERP_noaction[:,i])[1]
plt.plot(T_action,significance)
plt.hlines(0.05,-1000,500)

#%%% All ERPs

# Data
y=data
y=(y.T-y.mean(axis=-1)).T # demean
y=signal.detrend(y,axis=-1) # detrend
y=mne.filter.notch_filter(y,fs,np.arange(50, fs/2, 50),n_jobs=6)
l_freq=1
h_freq=30 # or 80 the result is the same
y=mne.filter.filter_data(y,fs,l_freq,h_freq,n_jobs=6)

### ERps aligned to image onset
Nt_trials=int((Pre+ISI+Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T=np.arange(-Npre,Npost)/fs*1000

ERP=np.zeros((Npics,Nc,Nt_trials))
for i in range(Npics):
     j=picstidx[i]
     erp=y[:,j-Npre:j+Npost]
     ERP[i]=erp

# Find deviating trials
ERPm=ERP.mean(axis=0)
ERPs=ERP.std(axis=0)
z=(ERP-ERPm)/ERPs
trialkeep=(((np.abs(z)<5).mean(axis=2))>0.97).all(axis=1)
print(trialkeep.sum(),trialkeep[trials1].sum(),trialkeep[trials0].sum())


### ERps aligned to joystick press / timeout
Nt_trials_action=int((Pre+Pre)*fs)
Npre_action=int(Pre*fs)
Npost_action=int((Pre)*fs)
T_action=np.arange(-Npre_action,Npost_action)/fs*1000

ERP_action=np.zeros((Npics,Nc,Nt_trials_action))
for i in range(Npics):
     j=endtrialtidx[i]
     erp=y[:,j-Npre_action:j+Npost_action]
     ERP_action[i]=erp


### Plots
for k in range(Nc):
    
    # Aligned to image
    erp=ERP[:,k,:]
    m=erp[:,:int(fs*0.4)].mean(axis=-1)
    s=erp[:,:int(fs*0.4)].std(axis=-1)
    #erp=((erp.T-m)/s).T
    erp=((erp.T-m)).T
        
    c=contacts[k]

    plt.figure(str(k)+'ERP',figsize=(10,6))
    
    m=np.median(erp[trials1][trialkeep[trials1]],axis=0)
    s=np.std(erp[trials1][trialkeep[trials1]],axis=0)
    plt.fill_between(T,m-s,m+s,color='green',alpha=0.2)
    plt.plot(T,m,label='Rec',color='green')
    
    m=np.median(erp[trials0][trialkeep[trials0]],axis=0)
    s=np.std(erp[trials0][trialkeep[trials0]],axis=0)
    plt.fill_between(T,m-s,m+s,color='red',alpha=0.2)
    plt.plot(T,m,label='Unrec',color='red')
    
    plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
    plt.title(contacts[k])
    
    plt.legend()
    plt.xlabel('time (ms)')
    plt.title('ERP '+whereDK[contacts[k]]+' '+str(whereMVV[contacts[k]]))
    plt.savefig(outputdir+'/ERP_'+c+'-Rec.png')
    plt.close()
    
    # Aligned to action / no action
    erp=ERP_action[:,k,:]
    m=erp[:,:int(fs*0.4)].mean(axis=-1)
    s=erp[:,:int(fs*0.4)].std(axis=-1)
    #erp=((erp.T-m)/s).T
    erp=((erp.T-m)).T
        
    c=contacts[k]

    plt.figure(str(k)+'Action ERP',figsize=(10,6))
    
    m=np.median(erp[trials1][trialkeep[trials1]],axis=0)
    s=np.std(erp[trials1][trialkeep[trials1]],axis=0)
    plt.fill_between(T_action,m-s,m+s,color='green',alpha=0.2)
    plt.plot(T_action,m,label='Rec',color='green')
    
    m=np.median(erp[trials0][trialkeep[trials0]],axis=0)
    s=np.std(erp[trials0][trialkeep[trials0]],axis=0)
    plt.fill_between(T_action,m-s,m+s,color='red',alpha=0.2)
    plt.plot(T_action,m,label='Unrec',color='red')
    
    plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
    plt.title(contacts[k])
    
    plt.legend()
    plt.xlabel('time (ms)')
    plt.title('Action ERP '+whereDK[contacts[k]]+' '+str(whereMVV[contacts[k]]))
    plt.savefig(outputdir+'/ERP_Action_'+c+'-Rec.png')
    plt.close()



#%% Connectivities broadbands

#%%% FC Pearson broadband

# Data
l_freq=5
h_freq=None # or 80 the result is the same
y=_data_filter(data, fs, l_freq, h_freq)

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000       
        

window_len=410 #int(fs/8)
window_step=1 #int(fs/64)
fs_FCbroad=int(2048/window_step)
Nw=int((Nt_trials)/window_step)


FC = np.zeros((Npics, Nc, Nc, Nw))
cov = np.zeros((Npics, Nc, Nc, Nw))
std = np.zeros((Npics, Nc, Nw))


# Epoch data
for i in range(Npics):
    j=picstidx[i]    
    data_epoch=y[:,j-Npre:j+Npost+window_len]  

    FC[i,:,:,:]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(Nw)]).swapaxes(0,2)
    std[i,:,:]=np.array([np.std(data_epoch[:,wi*window_step:wi*window_step+window_len],axis=-1) for wi in range(Nw)]).swapaxes(0,1)
    cov[i,:,:,:]=np.array([np.cov(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(Nw)]).swapaxes(0,2)
    





Tw=np.arange(Nw)*window_step/fs*1000-500
pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]


FCmean=np.zeros((Npics, Nw))
FCabsmean=np.zeros((Npics, Nw))
for i in range(Npics):
    for wi in range(Nw):
        FCmean[i,wi]=FC[i,K1s,K2s,wi].mean()
        FCabsmean[i,wi]=np.abs(FC[i,K1s,K2s,wi]).mean()
 
        
FCabs2mean=np.zeros((len(K1s), Nw))
covmean=np.zeros((len(K1s), Nw))
stdmean=np.zeros((Nc, Nw))
for wi in range(Nw):
    aux=FC[:,:,:,wi].mean(axis=0)
    FCabs2mean[:,wi]=np.abs(aux[K1s,K2s])
    aux=cov[:,:,:,wi].mean(axis=0)
    covmean[:,wi]=np.abs(aux[K1s,K2s])
    stdmean[:,wi]=std[:,:,wi].mean(axis=0)


vois=['FCmean','FCabsmean','FCabs2mean','covmean','stdmean']

for voiname in vois:
    
    voi=eval(voiname)
    NNN=voi.shape[0]

    if 'FC' in voiname:
        aux=fisherZ(voi)
        Zm=aux.mean(axis=0)
        #m=savgol_filter(m,411,polyorder=1)
        Zs=aux.std(axis=0) / np.sqrt(NNN)
        #s=savgol_filter(s,411,polyorder=1)
        m=fisherZinv(Zm)
        msu=fisherZinv(Zm+Zs)
        msl=fisherZinv(Zm-Zs)
    else:
        aux=voi
        m=aux.mean(axis=0)
        s=aux.std(axis=0) / np.sqrt(NNN)
        msu=m+s
        msl=m-s
    
    plt.figure(figsize=(10,6)) 
    plt.fill_between(Tw, msl, msu, color='k', alpha=0.2)
    plt.plot(Tw,m,color='k')
    #plt.hlines(hlines,T[0],T[-1],color='k',linestyle='dashed')        
    plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
    #plt.title(contacts[k])
    #plt.yticks(hlines,[-50,50]*Npics+[-20,20])
    plt.xlabel('')
    plt.ylabel('')
    #plt.title(voiname)
    xlabels=['' for i in np.arange(-500,1501,100)]
    xlabels[0]='-500'
    xlabels[5]='0'
    xlabels[10]='500'
    xlabels[15]='1000'
    xlabels[20]='1500'
    plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
    plt.yticks(fontsize=18)
    
    
    plt.savefig(outputdir+'/'+voiname+'_bandbroadZ.png')
    
    FCbroad=m
    np.save(outputdir+'/'+voiname+'_bandbroadZ.npy',FCbroad)




Tw=np.arange(Nw)*window_step/fs*1000-500
pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]
    
#preidx=(Tw<-100).nonzero()[0][-1]
win0=(Tw>300).nonzero()[0][0]
win1=(Tw<600).nonzero()[0][-1]
#[(FCmean[i,win0:win1]<FCmean[i,:preidx]).sum(axis=-1) for i in range(Npics)]
#[(FCmean[i,win0:win1]<FCmean[i,:preidx]).sum(axis=-1)/(preidx) for i in range(Npics)]
D0=FCabs2mean[:,win0:win1].mean(axis=-1)
D1=FCabs2mean[:,:pre_idx].mean(axis=-1)
print(st.wilcoxon(D0,D1))
print(st.ranksums(D0,D1))
#print(st.ttest_rel(D0,D1))
# # Epoch data
# FCmean=np.zeros((Npics, 2))
# for i in range(Npics):T
#     j=picstidx[i]
#     data_epoch=y[:,j-Npre:j+Npost]
#     fc=np.corrcoef(data_epoch[:,:int(fs/2)])
#     FCmean[i,0]=fc[K1s,K2s].mean()
#     fc=np.corrcoef(data_epoch[:,int(fs/2):3*int(fs/2)])
#     FCmean[i,1]=fc[K1s,K2s].mean()
# plt.plot(FCmean)

#%%% FC Pearson per bands

bandnames=['theta','alpha','beta','gamma','high-gamma']
bands=[(4,8),(8,12),(12,30),(30,70),(70,150)]
Nb=len(bands)

for bdi in range(Nb):
    
    bandname=bandnames[bdi]
    band=bands[bdi]
    
    # Data
    y=_data_filter(data, fs, band[0], band[1])  

    # Epoch data
    Nt_trials=int((Pre+Post)*fs)
    Npre=int(Pre*fs)
    Npost=int(Post*fs)
    T=np.arange(-Npre,Npost)/fs*1000       
        

    window_len=410#int(fs/8)
    window_step=41#int(fs/64)
    fs_FCbroad=int(2048/window_step)
    Nw=int((Nt_trials)/window_step)
    
    FC = np.zeros((Npics, Nc, Nc, Nw))
    
    # Epoch data
    for i in range(Npics):
        j=picstidx[i]        
        data_epoch=y[:,j-Npre:j+Npost+window_len]
        
        FC[i,:,:,:]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(Nw)]).swapaxes(0,2)
    
    Tw=np.arange(Nw)*window_step/fs*1000-500
    pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]
    FCmean=np.zeros((Npics, Nw))
    FCabsmean=np.zeros((Npics, Nw))
    for i in range(Npics):
        for wi in range(Nw):
            FCmean[i,wi]=FC[i,K1s,K2s,wi].mean()
            FCabsmean[i,wi]=np.abs(FC[i,K1s,K2s,wi]).mean()

    FCabs2mean=np.zeros((len(K1s), Nw))
    for wi in range(Nw):
        aux=FC[:,:,:,wi].mean(axis=0)
        FCabs2mean[:,wi]=np.abs(aux[K1s,K2s])
            
        #FCmean[i]=mne.filter.filter_data(FCmean[i],64,None,12,n_jobs=6)
        #m=FCmean[i,:preidx].mean(axis=-1)
        #s=FCmean[i,:preidx].std(axis=-1)
        #FCmean[i]=(FCmean[i]-m)
    


    vois=['FCmean','FCabsmean','FCabs2mean']
    
    for voiname in vois:
        
        voi=eval(voiname)
        NNN=voi.shape[0]
        
        aux=fisherZ(voi)
        Zm=aux.mean(axis=0)
        #m=savgol_filter(m,411,polyorder=1)
        Zs=aux.std(axis=0) / np.sqrt(NNN)
        #s=savgol_filter(s,411,polyorder=1)
        m=fisherZinv(Zm)
        msu=fisherZinv(Zm+Zs)
        msl=fisherZinv(Zm-Zs)
        
        plt.figure(figsize=(10,6)) 
        plt.fill_between(Tw, msl, msu, color='k', alpha=0.2)
        plt.plot(Tw,m,color='k')
        #plt.hlines(hlines,T[0],T[-1],color='k',linestyle='dashed')        
        plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
        #plt.title(contacts[k])
        #plt.yticks(hlines,[-50,50]*Npics+[-20,20])
        plt.xlabel('')
        plt.ylabel('')
        #plt.title(voiname)
        xlabels=['' for i in np.arange(-500,1501,100)]
        xlabels[0]='-500'
        xlabels[5]='0'
        xlabels[10]='500'
        xlabels[15]='1000'
        xlabels[20]='1500'
        plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.savefig(outputdir+'/'+voiname+'_band'+bandname+'.png')
        FCbroad=m
        np.save(outputdir+'/'+voiname+'_band'+bandname+'.npy',FCbroad)


#%%% PLV per bands

bandnames=['alpha','beta1','beta2','gamma1','gamma2','gamma3','gamma4']
bands=[(8,12),(12,18),(18,30),(30,45),(45,70),(70,100),(100,150)]
Nb=len(bands)

for bdi in range(Nb):
    
    bandname=bandnames[bdi]
    band=bands[bdi]
    
    # Data
    y=_data_filter(data, fs, band[0], band[1])
    angle=np.angle(signal.hilbert(y))

    # Epoch data
    Nt_trials=int((Pre+Post)*fs)
    Npre=int(Pre*fs)
    Npost=int(Post*fs)
    T=np.arange(-Npre,Npost)/fs*1000       
        
    
    window_len=410#int(fs/8)
    window_step=1#int(fs/64)
    fs_FCbroad=int(2048/window_step)
    Nw=int((Nt_trials)/window_step)
    
    PLVband = np.zeros((Npics, Nc, Nc, Nw))
    
 
    angle_epoch=np.zeros((Npics,Nc,Nt_trials+window_len))
    
    # Epoch data
    for i,i in enumerate(tqdm.tqdm(range(Npics), leave=False, desc='Trials')):
        j=picstidx[i]    
        angle_epoch[i]=angle[:,j-Npre:j+Npost+window_len]
                
    contact_results=Parallel(n_jobs=6)(delayed(_joblib_wrapper_cplv_singletrial)(angle_epoch,*ch_pair,window_len,window_step) for ch_pair in tqdm.tqdm(contactpairs, leave=False, desc='Edges'))   
        
    for pair, vec_cplv in zip(contactpairs, contact_results):
        vec_plv=np.abs(vec_cplv)
        PLVband[:,pair[0],pair[1],:]=vec_plv
 
    
 
    Tw=np.arange(Nw)*window_step/fs*1000-500
    pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]
    PLVmean=np.zeros((Npics, Nw))
    for i in range(Npics):
        for wi in range(Nw):
            PLVmean[i,wi]=PLVband[i,K1s,K2s,wi].mean()


    Cglobal=PLVmean
    Cglobal_norm=np.zeros((Npics,Nw))
    for i in range(Npics):
        m=Cglobal[i,:pre_idx].mean(axis=-1)
        Cglobal_norm[i]=Cglobal[i]-m 
    PLVmean_norm=Cglobal_norm
    
    
    vois=['PLVmean','PLVmean_norm']
    
    for voiname in vois:
        
        voi=eval(voiname)
        
        m=voi.mean(axis=0)
        #m=savgol_filter(m,411,polyorder=1)
        s=voi.std(axis=0)/np.sqrt(Npics)
        #s=savgol_filter(s,411,polyorder=1)
        #plt.figure(2)
        plt.fill_between(Tw, m-s, m+s, color='k', alpha=0.2)
        plt.plot(Tw,m)
        
        plt.figure(figsize=(10,6)) 
        plt.fill_between(Tw, m-s, m+s, color='k', alpha=0.2)
        plt.plot(Tw,m,'k')
        #plt.hlines(hlines,T[0],T[-1],color='k',linestyle='dashed')        
        plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
        #plt.title(contacts[k])
        #plt.yticks(hlines,[-50,50]*Npics+[-20,20])
        plt.xlabel('')
        plt.ylabel('')
        plt.title(voiname)
        xlabels=['' for i in np.arange(-500,1501,100)]
        xlabels[0]='-500'
        xlabels[5]='0'
        xlabels[10]='500'
        xlabels[15]='1000'
        xlabels[20]='1500'
        plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
        plt.yticks(fontsize=18)
        
        
        plt.savefig(outputdir+'/'+voiname+'_band'+bandname+'.png')
        
        FCbroad=m
        np.save(outputdir+'/'+voiname+'_band'+bandname+'.npy',voi)








#%%% FC strength per channel


# Data
l_freq=5
h_freq=None # or 80 the result is the same
y=_data_filter(data, fs, l_freq, h_freq)

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000       
        

window_len=410 #int(fs/8)
window_step=1 #int(fs/64)
fs_FCbroad=int(2048/window_step)
Nw=int((Nt_trials)/window_step)


FC = np.zeros((Npics, Nc, Nc, Nw))

# Epoch data
for i in range(Npics):
    j=picstidx[i]    
    data_epoch=y[:,j-Npre:j+Npost+window_len]  

    FC[i,:,:,:]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(Nw)]).swapaxes(0,2)
    
Tw=np.arange(Nw)*window_step/fs*1000-500
pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]
 
        
 
FC_strength=np.zeros((Nc, Nw))   
FCabs2=np.abs(FC.mean(axis=0))
for k in range(Nc):
    othercontacts=[k2 for (k1,k2) in contactpairs if k1==k]+[k1 for (k1,k2) in contactpairs if k2==k]
    aux=FCabs2[k,othercontacts,:]
    FC_strength[k,:]=np.mean(aux,axis=0)
    

m=FC_strength
m=(m.T-m[:,:pre_idx].mean(axis=1)).T
plt.figure('FC strength',figsize=(10,6))
plt.pcolormesh(Tw, range(Nc), m, cmap='jet', shading='none',vmin=-0.06,vmax=0.06)
xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
ylabels=[whereMVV[c] for c in contacts]
plt.yticks(range(Nc),ylabels,fontsize=9)
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.title('FC strength')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/FCstrength.png')
plt.close()   



#%%% FC strength in pre and post

win0=(Tw>300).nonzero()[0][0]
win1=(Tw<600).nonzero()[0][-1]

plt.figure(figsize=(10,6)) 


m=np.zeros(Nc)
s=np.zeros(Nc)
for k in range(Nc):
    othercontacts=[k2 for (k1,k2) in contactpairs if k1==k]+[k1 for (k1,k2) in contactpairs if k2==k]
    aux=FCabs2[k,othercontacts,:]
    FCinperiod=aux[:,:pre_idx].mean(axis=-1)
    m[k]=FCinperiod.mean(axis=0)
    s[k]=FCinperiod.std(axis=0)/np.sqrt(Nc)
#m=mm.mean(axis=0)
#s=mm.std(axis=0)/np.sqrt(Nc)
plt.fill_between(np.arange(Nc), m-s, m+s, color='b', alpha=0.2)
plt.plot(np.arange(Nc),m,label='baseline')


m=np.zeros(Nc)
s=np.zeros(Nc)
for k in range(Nc):
    othercontacts=[k2 for (k1,k2) in contactpairs if k1==k]+[k1 for (k1,k2) in contactpairs if k2==k]
    aux=FCabs2[k,othercontacts,:]
    FCinperiod=aux[:,win0:win1].mean(axis=-1)
    m[k]=FCinperiod.mean(axis=0)
    s[k]=FCinperiod.std(axis=0)/np.sqrt(Nc)
#mm=FC_strength[:,win0:win1].mean(axis=-1)
#m=mm.mean(axis=0)
#s=mm.std(axis=0)/np.sqrt(Nc)
plt.fill_between(np.arange(Nc), m-s, m+s, color='r', alpha=0.2)
plt.plot(np.arange(Nc),m,label='300-600 ms')

plt.legend()
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(outputdir+'/FC strenght pre post_.png')




#%%% FC in pre and post

# Data
l_freq=5
h_freq=None # or 80 the result is the same
y=_data_filter(data, fs, l_freq, h_freq)

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000       
        

window_len=410 #int(fs/8)
window_step=1 #int(fs/64)
fs_FCbroad=int(2048/window_step)
Nw=int((Nt_trials)/window_step)


FC = np.zeros((Npics, Nc, Nc, Nw))

# Epoch data
for i in range(Npics):
    j=picstidx[i]    
    data_epoch=y[:,j-Npre:j+Npost+window_len]  
    FC[i,:,:,:]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(Nw)]).swapaxes(0,2)
    
    
FCabs2=np.abs(FC.mean(axis=0))


#%%% FC in pre and post plot

plt.figure(figsize=(8,6)) 

Tw=np.arange(Nw)*window_step/fs*1000-500
pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]
FCabs2inperiod0=FCabs2[:,:,:pre_idx].mean(axis=-1)
win0=(Tw>300).nonzero()[0][0]
win1=(Tw<600).nonzero()[0][-1]
FCabs2inperiod1=FCabs2[:,:,win0:win1].mean(axis=-1)
plt.imshow(FCabs2inperiod1-FCabs2inperiod0,vmin=-0.1,vmax=0.1,origin='lower')
plt.colorbar()
ylabels=[whereMVV[c] for c in contacts]
plt.yticks(range(Nc),ylabels,fontsize=7)
plt.xticks(range(Nc),ylabels,fontsize=7,rotation=90)
plt.xlabel('contact ROI')
plt.ylabel('contact ROI')
plt.savefig(outputdir+'/FC post-pre.png')





#%%% FC Pearson aligned to button press

# Data
l_freq=5
h_freq=None # or 80 the result is the same
y=_data_filter(data, fs, l_freq, h_freq)


Nt_trials=int((2*Pre+Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000  

#np.concatenate((erp[:int(fs/2)],erp[-fs:]))   
        

window_len=410 #int(fs/8)
window_step=1 #int(fs/64)
fs_FCbroad=int(2048/window_step)
Nw=int((Nt_trials)/window_step)


FC1 = np.zeros((N1, Nc, Nc, Nw))
# Epoch data
for i in range(N1):
    trial=trials1[i]
    jpic=int(picstidx[trial])
    j=jpic
    #data_epoch=y[:,j-Npre:j+Npre]
    data_epoch=y[:,j-Npre:j+window_len]
    FC1[i,:,:,:int(fs/2)]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(int(fs/2))]).swapaxes(0,2)
    
    joystick=int(endtrialtidx[trial]) 
    j=joystick
    data_epoch=y[:,j-Npre:j+Npre+window_len]
    FC1[i,:,:,int(fs/2):]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(fs)]).swapaxes(0,2)



Tw=np.arange(Nw)*window_step/fs*1000-1000
pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]



FCmean1=np.zeros((N1, Nw))
FCabsmean1=np.zeros((N1, Nw))
for i in range(N1):
    for wi in range(Nw):
        FCmean1[i,wi]=np.abs(FC1[i,K1s,K2s,wi]).mean()
        FCabsmean1[i,wi]=np.abs(FC1[i,K1s,K2s,wi]).mean()
FCabs2mean1=np.zeros((len(K1s), Nw))
for wi in range(Nw):
    aux=FC1[:,:,:,wi].mean(axis=0)
    FCabs2mean1[:,wi]=np.abs(aux[K1s,K2s])
  
  
FC0 = np.zeros((N0, Nc, Nc, Nw))
for i in range(N0):
    trial=trials0[i]
    jpic=int(picstidx[trial])
    j=jpic
    #data_epoch=y[:,j-Npre:j+Npre]
    data_epoch=y[:,j-Npre+int(window_len/2):j+int(window_len/2)+window_len]
    FC0[i,:,:,:int(fs/2)]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(int(fs/2))]).swapaxes(0,2)
    
    joystick=int(endtrialtidx[trial]) 
    j=joystick
    data_epoch=y[:,j-Npre+int(window_len/2):j+Npre+int(window_len/2)+window_len]
    FC0[i,:,:,int(fs/2):]=np.array([np.corrcoef(data_epoch[:,wi*window_step:wi*window_step+window_len]) for wi in range(fs)]).swapaxes(0,2)


Tw=np.arange(Nw)*window_step/fs*1000-1000
pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]



FCmean0=np.zeros((N1, Nw))
FCabsmean0=np.zeros((N1, Nw))
for i in range(N1):
    for wi in range(Nw):
        FCmean0[i,wi]=np.abs(FC0[i,K1s,K2s,wi]).mean()
        FCabsmean0[i,wi]=np.abs(FC0[i,K1s,K2s,wi]).mean()
FCabs2mean0=np.zeros((len(K1s), Nw))
for wi in range(Nw):
    aux=FC0[:,:,:,wi].mean(axis=0)
    FCabs2mean0[:,wi]=np.abs(aux[K1s,K2s])


vois=['FCmean','FCabsmean','FCabs2mean']

for voiname in vois:
    
    plt.figure(figsize=(10,6))

    voi=eval(voiname+'0')
    NNN=voi.shape[0]
    aux=fisherZ(voi)
    Zm=aux.mean(axis=0)
    Zs=aux.std(axis=0) / np.sqrt(NNN)
    m=fisherZinv(Zm)
    msu=fisherZinv(Zm+Zs)
    msl=fisherZinv(Zm-Zs)
    plt.fill_between(Tw, msl, msu, color='r', alpha=0.2)
    plt.plot(Tw,m,'r',label='Unrecognized')
    np.save(outputdir+'/'+voiname+'_0_bandbroadZ.npy',m)
    
    voi=eval(voiname+'1')
    NNN=voi.shape[0]
    aux=fisherZ(voi)
    Zm=aux.mean(axis=0)
    Zs=aux.std(axis=0) / np.sqrt(NNN)
    m=fisherZinv(Zm)
    msu=fisherZinv(Zm+Zs)
    msl=fisherZinv(Zm-Zs)
    plt.fill_between(Tw, msl, msu, color='b', alpha=0.2)
    plt.plot(Tw,m,'b',label='Recognized')
    np.save(outputdir+'/'+voiname+'_1_bandbroadZ.npy',m)
    

    #plt.hlines(hlines,T[0],T[-1],color='k',linestyle='dashed')        
    plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
    #plt.title(contacts[k])
    #plt.yticks(hlines,[-50,50]*Npics+[-20,20])
    plt.xlabel('')
    plt.ylabel('')
    #plt.title(contacts[k]+voiname)
    plt.legend(fontsize=18)
    xlabels=['' for i in np.arange(-1000,501,100)]
    xlabels[6]='-400'
    xlabels[8]='-200'
    xlabels[10]='0'
    xlabels[12]='200'
    xlabels[14]='400'
    plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig(outputdir+'/'+voiname+'_01_bandbroadZ.png')


    


    


significance=np.zeros(int(1.5*fs))
for i in range(significance.shape[0]):
    significance[i]=st.ranksums(FCabs2mean0[:,i],FCabs2mean1[:,i])[1]
plt.plot(T_action,significance)
plt.hlines(0.05,-1000,500)




    
#preidx=(Tw<-100).nonzero()[0][-1]
win0=(Tw>-500).nonzero()[0][0]
win1=(Tw<100).nonzero()[0][-1]
#[(FCmean[i,win0:win1]<FCmean[i,:preidx]).sum(axis=-1) for i in range(Npics)]
#[(FCmean[i,win0:win1]<FCmean[i,:preidx]).sum(axis=-1)/(preidx) for i in range(Npics)]
D0=FCabs2mean0[:,win0:win1].mean(axis=-1)
D1=FCabs2mean1[:,win0:win1].mean(axis=-1)
print(st.wilcoxon(D0,D1))
print(st.ranksums(D0,D1))


#[(FCmean[i,74]<FCmean[i,:preidx]).sum(axis=-1) for i in range(Npics)]
#[(FCmean[i,74]<FCmean[i,:preidx]).sum(axis=-1)/(preidx) for i in range(Npics)]
#D0=list(FCmean[:,74])
#D1=list(FCmean[:,:preidx].flatten())
#st.ranksums(D0,D1)

#%% MULTITAPER


#%%% Multitaper power

# Data
y=data
y=(y.T-y.mean(axis=-1)).T # demean
y=signal.detrend(y,axis=-1) # detrend
y=mne.filter.notch_filter(y,fs,np.arange(50, fs/2, 50),n_jobs=6)
l_freq=1
h_freq=700
y=mne.filter.filter_data(y,fs,l_freq,h_freq,n_jobs=6)

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000


# Prepare Multitaper parameters
def get_frequencies():
    return 2**np.arange(1,np.log2(512+1),1/4)
frequencies=get_frequencies()
Nf=frequencies.shape[0]
W=(frequencies*2**(3/4)-frequencies)/2
#W=freq/2+0.001

fcut=32

# For f<32 Hz, I want a single taper. Therefore np.floor(2TW-1)=1
W1=W[frequencies<fcut]
T1=1/W1
#ncycles1=np.ceil(T1*frequencies[frequencies<fcut])
ncycles1=6*np.ones(T1.shape[0])
T1=(ncycles1/frequencies[frequencies<fcut])
ntaps1=np.floor(2*T1*W1-1)

# For f>32 Hz
W2=W[frequencies>=fcut]
T2=0.200
ncycles2=np.ceil(T2*frequencies[frequencies>=fcut])
T2=(ncycles2/frequencies[frequencies>=fcut])
ntaps2=np.floor(2*T2*W2-1)

# Join
ncycles=np.concatenate((ncycles1,ncycles2))
ntaps=np.concatenate((ntaps1,ntaps2))
TT=np.concatenate((T1,T2))
FullTW=2*TT*W


cwtm=np.zeros((Nf,Nc,Nt))
for fi, freq in enumerate(tqdm.tqdm(frequencies, leave=False, desc='Frequencies')):
    data_preprocessed=mne.time_frequency.tfr_array_multitaper(y[np.newaxis,...], int(fs), [freq], n_cycles=ncycles[fi],time_bandwidth=FullTW[fi],output='power', n_jobs=6).squeeze()
    cwtm[fi,:,:] = data_preprocessed


#k=0 #OC1
#print(k,end='')
#c=contacts[k]

# Continuous multitaper decomposition
# cwtm=np.zeros((Nfreq,Nc,int(Nt/1)))

# for (i,f) in enumerate(freq):
#     print (i,f)
#     out=mne.time_frequency.tfr_array_multitaper(y[np.newaxis,:,:],sfreq=int(fs),freqs=[f],n_cycles=ncycles[i],time_bandwidth=FullTW[i],output='power')
#     cwtm[i,:,:]=out[0,:,0,:]

print('done')


#%%% Save spectrograms

for k in range(Nc):
    print(k)
    np.save(outputdir+'/cwtm_multitaper_ch'+str(k)+'.npy',cwtm[:,k,:])
    
    
#%%% Load spectrograms

cwtm=np.zeros((Nf,Nc,Nt))
for k in range(Nc):
    print(k)
    cwtm[:,k,:]=np.load(outputdir+'/cwtm_multitaper_ch'+str(k)+'.npy')
    

#%%% Epoch data and save

freq=2**np.arange(1,np.log2(512+1),1/4)

# Normalization startegy
norm='mean'

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000


for k in range(Nc):
    print(k,end='')
    c=contacts[k]
    cwtm_trials=np.zeros((Npics,Nf,int(Nt_trials)))
    
    for i in range(Npics):
         j=int(picstidx[i])
         yy=cwtm[:,k,j-Npre:j+Npost]
         m=yy[:,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean(axis=-1)
         if norm=='mean':
             cwtm_trials[i]=(yy.T/m).T

    m=np.median(cwtm_trials,axis=0)
    np.save(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'.npy',m)
    

#%%% Plot spectrograms

freq=2**np.arange(1,np.log2(512+1),1/4)
Nf=freq.shape[0]

# Normalization startegy
norm='mean'

# time points : PRE (500 ms), ISI (800 ms aprox), POST (we use Pre, 500ms)
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs/1)
Npost=int((Post)*fs/1)
T=np.arange(-Npre,Npost)/fs*1000

for k in range(Nc):
    
    #k=0 #OC1
    print(k,end='')
    c=contacts[k]
    cwtm_trials1=np.zeros((N1,Nf,int(Nt_trials)))
    for i in range(N1):
         trial=trials1[i]
         j=int(picstidx[trial])
         yy=cwtm[:,k,j-Npre:j+Npost]
         m=yy[:,0:Npre].mean(axis=-1)
         cwtm_trials1[i]=(yy.T/m).T
         
    cwtm_trials0=np.zeros((N0,Nf,int(Nt_trials)))
    for i in range(N0):
         trial=trials0[i]
         j=int(picstidx[trial])
         yy=cwtm[:,k,j-Npre:j+Npost]
         m=yy[:,0:Npre].mean(axis=-1)
         cwtm_trials0[i]=(yy.T/m).T
         
    ### Figure
    m=np.median(cwtm_trials1,axis=0)
    vmax=1.5
    vmin=0.5
    plt.figure(str(k)+'average spectrogram',figsize=(10,6))
    plt.pcolormesh(T, range(Nf), m, vmin=vmin, vmax=vmax, cmap='jet', shading='gouraud')
    ylabels=[int(round(freq[i],0)) for i in range(0,Nf,4)]
    plt.yticks(range(len(freq))[::4],ylabels)
    plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
    plt.title(whereDK[contacts[k]]+' '+str(whereMVV[contacts[k]]))
    plt.colorbar()
    plt.savefig(outputdir+'/Spectrogram_multitaper_'+c+'_rec.png')
    plt.close() 
    
    m=np.median(cwtm_trials0,axis=0)
    vmax=1.5
    vmin=0.5
    plt.figure(str(k)+'average spectrogram',figsize=(10,6))
    plt.pcolormesh(T, range(Nf), m, vmin=vmin, vmax=vmax, cmap='jet', shading='gouraud')
    ylabels=[int(round(freq[i],0)) for i in range(0,Nf,4)]
    plt.yticks(range(len(freq))[::4],ylabels)
    plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
    plt.title(whereDK[contacts[k]]+' '+str(whereMVV[contacts[k]]))
    plt.colorbar()
    plt.savefig(outputdir+'/Spectrogram_multitaper_'+c+'_nonrec.png')
    plt.close() 
    
    
#%%% Plot average time-frequency energy for paper


freq=2**np.arange(1,np.log2(512+1),1/4)
Nf=freq.shape[0]

# Normalization startegy
eachtrial=0
norm='mean'

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000


cwtm_av=np.zeros((Nf,Nt))
for k in range(Nc):
    print(k)
    cwtm_av[:,:]+=np.load(outputdir+'/cwtm_multitaper_ch'+str(k)+'.npy')
cwtm_av/=Nc
    
cwtm_trials=np.zeros((Npics,Nf,int(Nt_trials)))

for i in range(Npics):
     j=int(picstidx[i])
     yy=cwtm_av[:,j-Npre:j+Npost]
     m=yy[:,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean(axis=-1)
     if norm=='mean':
         cwtm_trials[i]=(yy.T/m).T

### Figure
m=np.median(cwtm_trials,axis=0)
m=m[freq>=4]
vmax=np.max(m.max())
vmin=np.min(m.min())

M=max(abs(vmax),abs(vmin))
vmax=1.5#np.percentile(M,95)
vmin=0.5

freq=freq[freq>=4]
Nf=freq.shape[0]

plt.figure(str(k)+'average spectrogram',figsize=(10,6))
plt.pcolormesh(T, range(Nf), m, vmin=vmin, vmax=vmax, cmap='jet', shading='gouraud')
xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
ylabels=[int(round(freq[fi],0)) for fi in range(0,Nf,4)]
plt.yticks(range(len(freq))[::4],ylabels,fontsize=18)
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.title('')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/Average_Spectrogram_multitaper.png')
plt.close()

    
#%%% Plot spectrograms aligned to response

def get_frequencies():
    return 2**np.arange(1,np.log2(512+1),1/4)
freq=get_frequencies()
Nf=freq.shape[0]

# Normalization startegy
norm='mean'

# time points : PRE (500 ms), ISI (800 ms aprox), POST (we use Pre, 500ms)

Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000


for k in range(Nc):
    freq=get_frequencies()
    Nf=freq.shape[0]
    
    #k=0 #OC1
    print(k,end='')
    c=contacts[k]
    cwtm_trials1=np.zeros((N1,Nf,Nt_trials_action))
    for i in range(N1):        
         trial=trials1[i]
         j=int(picstidx[trial])
         joystick=int(endtrialtidx[trial])   
         yy=cwtm[:,k,j-Npre:joystick+Npre]
         m=yy[:,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean(axis=-1)
         yy=(yy.T/m).T
         cwtm_trials1[i]=np.concatenate((yy[:,:int(fs/2)],yy[:,-fs:]),axis=-1)
         
    cwtm_trials0=np.zeros((N0,Nf,Nt_trials_action))
    for i in range(N0):
         trial=trials0[i]
         j=int(picstidx[trial])
         joystick=int(endtrialtidx[trial])   
         yy=cwtm[:,k,j-Npre:joystick+Npre]
         m=yy[:,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean(axis=-1)
         yy=(yy.T/m).T
         cwtm_trials0[i]=np.concatenate((yy[:,:int(fs/2)],yy[:,-fs:]),axis=-1)
         
    ### Figure
    m=np.median(cwtm_trials1,axis=0)
    m=m[freq>=4]
    freq=freq[freq>=4]
    Nf=freq.shape[0]
    vmax=1.5
    vmin=0.5
    plt.figure(str(k)+'average spectrogram',figsize=(10,6))
    plt.pcolormesh(T_action, range(Nf), m, vmin=vmin, vmax=vmax, cmap='jet', shading='gouraud')
    xlabels=['' for i in np.arange(-1000,501,100)]
    #xlabels[1]='-400'
    #xlabels[3]='-200'
    #xlabels[5]=''
    xlabels[6]='-400'
    xlabels[8]='-200'
    xlabels[10]='0'
    xlabels[12]='200'
    xlabels[14]='400'
    plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
    ylabels=[int(round(freq[fi],0)) for fi in range(0,Nf,4)]
    plt.yticks(range(len(freq))[::4],ylabels,fontsize=18)
    
    ylabels=[int(round(freq[i],0)) for i in range(0,Nf,4)]
    plt.yticks(range(len(freq))[::4],ylabels)
    #plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='white',alpha=0.5,linewidth=3)
    plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
    plt.title('')
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(outputdir+'/res_Spectrogram_multitaper_'+c+'_rec.png')
    plt.close() 
    
    freq=get_frequencies()
    Nf=freq.shape[0]
    m=np.median(cwtm_trials0,axis=0)
    m=m[freq>=4]
    freq=freq[freq>=4]
    Nf=freq.shape[0]
    vmax=1.5
    vmin=0.5
    plt.figure(str(k)+'average spectrogram',figsize=(10,6))
    plt.pcolormesh(T_action, range(Nf), m, vmin=vmin, vmax=vmax, cmap='jet', shading='gouraud')
    xlabels=['' for i in np.arange(-1000,501,100)]
    xlabels[6]='-400'
    xlabels[8]='-200'
    xlabels[10]='0'
    xlabels[12]='200'
    xlabels[14]='400'
    plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
    ylabels=[int(round(freq[i],0)) for i in range(0,Nf,4)]
    plt.yticks(range(len(freq))[::4],ylabels,fontsize=18)
    #plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='white',alpha=0.5,linewidth=3)
    plt.title('')
    plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.savefig(outputdir+'/res_Spectrogram_multitaper_'+c+'_nonrec.png')
    plt.close()  
    
    
    
#%%% Epoch data and save aligned to response

def get_frequencies():
    return 2**np.arange(1,np.log2(512+1),1/4)
freq=get_frequencies()

# Normalization startegy
norm='mean'

Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000


for k in range(Nc):
    freq=get_frequencies()
    Nf=freq.shape[0]
    
    #k=0 #OC1
    print(k,end='')
    c=contacts[k]
    cwtm_trials1=np.zeros((N1,Nf,Nt_trials_action))
    for i in range(N1):        
         trial=trials1[i]
         j=int(picstidx[trial])
         joystick=int(endtrialtidx[trial])   
         yy=cwtm[:,k,j-Npre:joystick+Npre]
         m=yy[:,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean(axis=-1)
         yy=(yy.T/m).T
         cwtm_trials1[i]=np.concatenate((yy[:,:int(fs/2)],yy[:,-fs:]),axis=-1)
         
    cwtm_trials0=np.zeros((N0,Nf,Nt_trials_action))
    for i in range(N0):
         trial=trials0[i]
         j=int(picstidx[trial])
         joystick=int(endtrialtidx[trial])   
         yy=cwtm[:,k,j-Npre:joystick+Npre]
         m=yy[:,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean(axis=-1)
         yy=(yy.T/m).T
         cwtm_trials0[i]=np.concatenate((yy[:,:int(fs/2)],yy[:,-fs:]),axis=-1)
         
    
    m=np.median(cwtm_trials1,axis=0)
    np.save(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'_rec.npy',m)

    m=np.median(cwtm_trials0,axis=0)
    np.save(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'_nonrec.npy',m)

    


#%%% TEST for selected TF windows

import scipy.stats as st

norm='mean'

# Region 1: M1, contact FP13 
c='FP13'
Fwin=[90,128]
Twin=[1000,1200]



k=contacts.index(c)

fidx=((freq>=Fwin[0])*(freq<=Fwin[1])).nonzero()[0]

tidx=((T>=Twin[0])*(T<=Twin[1])).nonzero()[0]

Tdur=Twin[1]-Twin[0]
delta0=(500-Tdur)/2
Twinpre=[-500+delta0,-500+delta0+Tdur]
tpreidx=((T>=Twinpre[0])*(T<=Twinpre[1])).nonzero()[0]



cwtm_trials=np.zeros((N1,Nfreq,int(Nt_trials)))
for i in range(N1):
     trial=trials1[i]
     j=int(picstidx[trial])
     yy=cwtm[:,k,j-Npre:j+Npost]
     #yy=signal.savgol_filter(yyaux,window_length=101,polyorder=0,axis=-1)
     m=yy[:,0:Npre].mean(axis=-1)
     mt=yy.mean(axis=-1)
     s=yy[:,:Npre].std(axis=-1)
     M=cwtm[:,k,:].mean(axis=-1)
     #s=(np.abs(cwtm[:,1*fs:7*fs])**2).mean(axis=-1)
     if norm=='mean':
         cwtm_trials[i]=(yy.T/m).T
     elif norm=='meantotal':
         cwtm_trials[i]=(yy.T/M).T
     elif norm=='meantrial':
         cwtm_trials[i]=(yy.T/mt).T
     elif norm=='zscore':
         cwtm_trials[i]=((yy.T-m)/s).T
     elif norm=='none':
         cwtm_trials[i]=yy
         
D1=np.zeros(N1)
for i in range(N1):
    D1[i]=cwtm_trials[i][fidx,:][:,tidx].mean()
D0=np.zeros(N1)
for i in range(N1):
    D0[i]=cwtm_trials[i][fidx,:][:,tpreidx].mean()
    

print(st.ranksums(D1,D0))


#%%% DLPFC, contact FM12

c='FM12'
Fwin=[16,32]
Twin=[500,750]

k=contacts.index(c)

fidx=((freq>=Fwin[0])*(freq<=Fwin[1])).nonzero()[0]

tidx=((T>=Twin[0])*(T<=Twin[1])).nonzero()[0]

Tdur=Twin[1]-Twin[0]
delta0=(500-Tdur)/2
Twinpre=[-500+delta0,-500+delta0+Tdur]
tpreidx=((T>=Twinpre[0])*(T<=Twinpre[1])).nonzero()[0]



cwtm_trials=np.zeros((N1,Nfreq,int(Nt_trials)))
for i in range(N1):
     trial=trials1[i]
     j=int(picstidx[trial])
     yy=cwtm[:,k,j-Npre:j+Npost]
     #yy=signal.savgol_filter(yyaux,window_length=101,polyorder=0,axis=-1)
     m=yy[:,0:Npre].mean(axis=-1)
     mt=yy.mean(axis=-1)
     s=yy[:,:Npre].std(axis=-1)
     M=cwtm[:,k,:].mean(axis=-1)
     #s=(np.abs(cwtm[:,1*fs:7*fs])**2).mean(axis=-1)
     if norm=='mean':
         cwtm_trials[i]=(yy.T/m).T
     elif norm=='meantotal':
         cwtm_trials[i]=(yy.T/M).T
     elif norm=='meantrial':
         cwtm_trials[i]=(yy.T/mt).T
     elif norm=='zscore':
         cwtm_trials[i]=((yy.T-m)/s).T
     elif norm=='none':
         cwtm_trials[i]=yy
         
D1=np.zeros(N1)
for i in range(N1):
    D1[i]=cwtm_trials[i][fidx,:][:,tidx].mean()
D0=np.zeros(N1)
for i in range(N1):
    D0[i]=cwtm_trials[i][fidx,:][:,tpreidx].mean()
    

print(st.ranksums(D1,D0))


#%%% DLPFC, contact FA10
c='FA10'
Fwin=[64,90]
Twin=[450,550]




k=contacts.index(c)

fidx=((freq>=Fwin[0])*(freq<=Fwin[1])).nonzero()[0]

tidx=((T>=Twin[0])*(T<=Twin[1])).nonzero()[0]

Tdur=Twin[1]-Twin[0]
delta0=(500-Tdur)/2
Twinpre=[-500+delta0,-500+delta0+Tdur]
tpreidx=((T>=Twinpre[0])*(T<=Twinpre[1])).nonzero()[0]



cwtm_trials=np.zeros((N1,Nfreq,int(Nt_trials)))
for i in range(N1):
     trial=trials1[i]
     j=int(picstidx[trial])
     yy=cwtm[:,k,j-Npre:j+Npost]
     #yy=signal.savgol_filter(yyaux,window_length=101,polyorder=0,axis=-1)
     m=yy[:,0:Npre].mean(axis=-1)
     mt=yy.mean(axis=-1)
     s=yy[:,:Npre].std(axis=-1)
     M=cwtm[:,k,:].mean(axis=-1)
     #s=(np.abs(cwtm[:,1*fs:7*fs])**2).mean(axis=-1)
     if norm=='mean':
         cwtm_trials[i]=(yy.T/m).T
     elif norm=='meantotal':
         cwtm_trials[i]=(yy.T/M).T
     elif norm=='meantrial':
         cwtm_trials[i]=(yy.T/mt).T
     elif norm=='zscore':
         cwtm_trials[i]=((yy.T-m)/s).T
     elif norm=='none':
         cwtm_trials[i]=yy
         
D1=np.zeros(N1)
for i in range(N1):
    D1[i]=cwtm_trials[i][fidx,:][:,tidx].mean()
D0=np.zeros(N1)
for i in range(N1):
    D0[i]=cwtm_trials[i][fidx,:][:,tpreidx].mean()
    

print(st.ranksums(D1,D0))


#%%% TEST for selected TF windows aligned to response

import scipy.stats as st
def get_frequencies():
    return 2**np.arange(1,np.log2(512+1),1/4)
freq=get_frequencies()
Nf=freq.shape[0]
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000

norm='mean'

# c='FP13'
# Fwin=[64,128]
# Twin=[-50,200]

# c='FP13'
# Fwin=[16,32]
# Twin=[-500,-250]


c='FM12'
Fwin=[16,32]
Twin=[-500,-250]



k=contacts.index(c)

fidx=((freq>=Fwin[0])*(freq<=Fwin[1])).nonzero()[0]

tidx=((T_action>=Twin[0])*(T_action<=Twin[1])).nonzero()[0]

Tdur=Twin[1]-Twin[0]
delta0=(500-Tdur)/2
Twinpre=[-1000+delta0,-1000+delta0+Tdur]
tpreidx=((T_action>=Twinpre[0])*(T_action<=Twinpre[1])).nonzero()[0]

cwtm_trials1=np.zeros((N1,Nf,Nt_trials_action))
for i in range(N1):        
     trial=trials1[i]
     j=int(picstidx[trial])
     joystick=int(endtrialtidx[trial])   
     yy=cwtm[:,k,j-Npre:joystick+Npre]
     m=yy[:,0:Npre].mean(axis=-1)
     yy=(yy.T/m).T
     cwtm_trials1[i]=np.concatenate((yy[:,:int(fs/2)],yy[:,-fs:]),axis=-1)
     
cwtm_trials0=np.zeros((N0,Nf,Nt_trials_action))
for i in range(N0):
     trial=trials0[i]
     j=int(picstidx[trial])
     joystick=int(endtrialtidx[trial])   
     yy=cwtm[:,k,j-Npre:joystick+Npre]
     m=yy[:,0:Npre].mean(axis=-1)
     yy=(yy.T/m).T
     cwtm_trials0[i]=np.concatenate((yy[:,:int(fs/2)],yy[:,-fs:]),axis=-1)
         
D1=np.zeros(N1)
for i in range(N1):
    D1[i]=cwtm_trials1[i][fidx,:][:,tidx].mean()
Dpre=np.zeros(N1) 
for i in range(N1):
    Dpre[i]=cwtm_trials1[i][fidx,:][:,tpreidx].mean()
D0=np.zeros(N0)
for i in range(N0):
    D0[i]=cwtm_trials0[i][fidx,:][:,tidx].mean()

    

print(st.ranksums(D1,D0)) 

print(st.ranksums(D1,Dpre)) 


#%% Multitaper phase

#%%% Estimate phases

# Data
y=data
y=(y.T-y.mean(axis=-1)).T # demean
y=signal.detrend(y,axis=-1) # detrend
y=mne.filter.notch_filter(y,fs,np.arange(50, fs/2, 50),n_jobs=6)
l_freq=1
h_freq=700
y=mne.filter.filter_data(y,fs,l_freq,h_freq,n_jobs=6)

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000


# Prepare Multitaper parameters
def get_frequencies():
    return 2**np.arange(1,np.log2(512+1),1/4)
frequencies=get_frequencies()
Nf=frequencies.shape[0]
W=(frequencies*2**(3/4)-frequencies)/2
#W=freq/2+0.001

fcut=32

# For f<32 Hz, I want a single taper. Therefore np.floor(2TW-1)=1
W1=W[frequencies<fcut]
T1=1/W1
#ncycles1=np.ceil(T1*frequencies[frequencies<fcut])
ncycles1=6*np.ones(T1.shape[0])
T1=(ncycles1/frequencies[frequencies<fcut])
ntaps1=np.floor(2*T1*W1-1)

# For f>32 Hz
W2=W[frequencies>=fcut]
T2=0.200
ncycles2=np.ceil(T2*frequencies[frequencies>=fcut])
T2=(ncycles2/frequencies[frequencies>=fcut])
ntaps2=np.floor(2*T2*W2-1)

# Join
ncycles=np.concatenate((ncycles1,ncycles2))
ntaps=np.concatenate((ntaps1,ntaps2))
TT=np.concatenate((T1,T2))
FullTW=2*TT*W



# Continuous multitaper decomposition
phase=np.zeros((Nf,Nc,Nt))

for fi, freq in enumerate(tqdm.tqdm(frequencies, leave=False, desc='Frequencies')):
    data_preprocessed=mne.time_frequency.tfr_array_multitaper(y[np.newaxis,...], int(fs), [freq], n_cycles=ncycles[fi],time_bandwidth=FullTW[fi],output='complex', n_jobs=6).squeeze()
    phase[fi,:,:] = np.angle(data_preprocessed)


print('done')


#%%% Save phases

for k in range(Nc):
    print(k)
    np.save(outputdir+'/phasesmultitaper_ch'+str(k)+'.npy',phase[:,k,:])


#%%% Load phases

phase=np.zeros((Nf,Nc,Nt))
for k in range(Nc):
    print(k)
    phase[:,k,:]=np.load(outputdir+'/phasesmultitaper_ch'+str(k)+'.npy')
    



#%% mPLV

#%%% Estimate mPLV

frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
phase=np.zeros((Nf,Nc,Nt))
for k in range(Nc):
    phase[:,k,:]=np.load(outputdir+'/phasesmultitaper_ch'+str(k)+'.npy')

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000

phase_epochs=np.zeros((Npics,Nf,Nc,Nt_trials))
for i in range(Npics):
    j=picstidx[i]
    phase_epochs[i]=phase[:,:,j-Npre:j+Npost] # phase=np.zeros((Nfreq,Nc,Nt))
    
    
PLVt=np.zeros((Nf,Nc,Nc,Nt_trials))
for (k1,k2) in tqdm.tqdm(contactpairs):
    PLVt[:,k1,k2,:]=np.abs((np.exp((phase_epochs[:,:,k1,:]-phase_epochs[:,:,k2,:])*1j)).mean(axis=0))



mPLV=np.zeros((Nf,Nt_trials))
for fi in range(Nf):
    for ti in range(Nt_trials):
        mPLV[fi,ti]=PLVt[fi,K1s,K2s,ti].mean()
    #rtm=PLVmean[fi,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean()
    #PLVmean[fi,:]=PLVmean[fi,:]-rtm

np.save(outputdir+'/mPLV.npy',mPLV)

#%%% Plot mPLV

mPLV=np.load(outputdir+'/mPLV.npy')
(Nf,Nt_trials)=mPLV.shape
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000
frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
m=mPLV


### Figure
m=m[frequencies>=4]
freq=frequencies[frequencies>=4]
Nf=freq.shape[0]

plt.figure('PLVmean',figsize=(12,6))

norm_idx=(T<-100).nonzero()[0][-1]
m=((m.T-m[:,:norm_idx].mean(axis=-1))/m[:,:norm_idx].std(axis=-1)).T
plt.pcolormesh(T, range(Nf), m, vmin=-4, vmax=4, cmap='jet')

#m=((m.T-m[:,:pre_idx].mean(axis=-1))).T
#plt.pcolormesh(T, range(Nf), m, vmin=-0.05, vmax=0.05, cmap='jet')

xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
ylabels=[int(round(freq[fi],0)) for fi in range(0,Nf,4)]
plt.yticks(range(len(freq))[::4],ylabels,fontsize=18)
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.title('')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/PLVmean_paper.png')
plt.close()



#%%% Estimate mPLV aligned to response REC

frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
phase=np.zeros((Nf,Nc,Nt))
for k in range(Nc):
    phase[:,k,:]=np.load(outputdir+'/phasesmultitaper_ch'+str(k)+'.npy')
    
# time points : PRE (500 ms), ISI (800 ms aprox), POST (we use Pre, 500ms)
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
         
phase_epochs=np.zeros((N1,Nf,Nc,Nt_trials_action))
for i in range(N1):
    trial=trials1[i]
    j=int(picstidx[trial])
    joystick=int(endtrialtidx[trial]) 
    yy=phase[:,:,j-Npre:joystick+Npre]
    phase_epochs[i]=np.concatenate((yy[:,:,:int(fs/2)],yy[:,:,-fs:]),axis=-1)    
PLVt=np.zeros((Nf,Nc,Nc,Nt_trials_action))
for (k1,k2) in tqdm.tqdm(contactpairs):
    PLVt[:,k1,k2,:]=np.abs((np.exp((phase_epochs[:,:,k1,:]-phase_epochs[:,:,k2,:])*1j)).mean(axis=0))
    
mPLV=np.zeros((Nf,Nt_trials_action))
for fi in range(Nf):
    for ti in range(Nt_trials_action):
        mPLV[fi,ti]=PLVt[fi,K1s,K2s,ti].mean()
    #rtm=PLVmean[fi,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean()
    #PLVmean[fi,:]=PLVmean[fi,:]-rtm

np.save(outputdir+'/mPLV1.npy',mPLV)
    
#%%% Plot mPLV REC

mPLV=np.load(outputdir+'/mPLV1.npy')
(Nf,Nt_trials_action)=mPLV.shape
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
m=mPLV

### Figure
m=m[frequencies>=4]
freq=frequencies[frequencies>=4]
Nf=freq.shape[0]

plt.figure('PLVmean1',figsize=(12,6))

norm_idx=(T_action<-600).nonzero()[0][-1]
m=((m.T-m[:,:norm_idx].mean(axis=-1))/m[:,:norm_idx].std(axis=-1)).T
plt.pcolormesh(T_action, range(Nf), m, vmin=-4, vmax=4, cmap='jet')

#m=((m.T-m[:,:pre_idx].mean(axis=-1))).T
#plt.pcolormesh(T, range(Nf), m, vmin=-0.05, vmax=0.05, cmap='jet')

xlabels=['' for i in np.arange(-1000,501,100)]
xlabels[6]='-400'
xlabels[8]='-200'
xlabels[10]='0'
xlabels[12]='200'
xlabels[14]='400'
plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
ylabels=[int(round(freq[fi],0)) for fi in range(0,Nf,4)]
plt.yticks(range(len(freq))[::4],ylabels,fontsize=18)
ylabels=[int(round(freq[i],0)) for i in range(0,Nf,4)]
plt.yticks(range(len(freq))[::4],ylabels)
#plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='white',alpha=0.5,linewidth=3)
plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.title('')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/PLVmean1_paper.png')
plt.close()


#%%% Estimate mPLV aligned to response NONREC

frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
phase=np.zeros((Nf,Nc,Nt))
for k in range(Nc):
    phase[:,k,:]=np.load(outputdir+'/phasesmultitaper_ch'+str(k)+'.npy')
    
# time points : PRE (500 ms), ISI (800 ms aprox), POST (we use Pre, 500ms)
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
         
phase_epochs=np.zeros((N0,Nf,Nc,Nt_trials_action))
for i in range(N0):
    trial=trials0[i]
    j=int(picstidx[trial])
    joystick=int(endtrialtidx[trial]) 
    yy=phase[:,:,j-Npre:joystick+Npre]
    phase_epochs[i]=np.concatenate((yy[:,:,:int(fs/2)],yy[:,:,-fs:]),axis=-1)    
PLVt=np.zeros((Nf,Nc,Nc,Nt_trials_action))
for (k1,k2) in tqdm.tqdm(contactpairs):
    PLVt[:,k1,k2,:]=np.abs((np.exp((phase_epochs[:,:,k1,:]-phase_epochs[:,:,k2,:])*1j)).mean(axis=0))
    
mPLV=np.zeros((Nf,Nt_trials_action))
for fi in range(Nf):
    for ti in range(Nt_trials_action):
        mPLV[fi,ti]=PLVt[fi,K1s,K2s,ti].mean()
    #rtm=PLVmean[fi,int(2048*100/1000):int(Npre)-int(2048*100/1000)].mean()
    #PLVmean[fi,:]=PLVmean[fi,:]-rtm

np.save(outputdir+'/mPLV0.npy',mPLV)


#%%% Plot mPLV NONREC

mPLV=np.load(outputdir+'/mPLV0.npy')
(Nf,Nt_trials_action)=mPLV.shape
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
m=mPLV

### Figure
m=m[frequencies>=4]
freq=frequencies[frequencies>=4]
Nf=freq.shape[0]

plt.figure('PLVmean0',figsize=(12,6))

norm_idx=(T_action<-600).nonzero()[0][-1]
m=((m.T-m[:,:norm_idx].mean(axis=-1))/m[:,:norm_idx].std(axis=-1)).T
plt.pcolormesh(T_action, range(Nf), m, vmin=-4, vmax=4, cmap='jet')

#m=((m.T-m[:,:pre_idx].mean(axis=-1))).T
#plt.pcolormesh(T, range(Nf), m, vmin=-0.05, vmax=0.05, cmap='jet')

xlabels=['' for i in np.arange(-1000,501,100)]
xlabels[6]='-400'
xlabels[8]='-200'
xlabels[10]='0'
xlabels[12]='200'
xlabels[14]='400'
plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
ylabels=[int(round(freq[fi],0)) for fi in range(0,Nf,4)]
plt.yticks(range(len(freq))[::4],ylabels,fontsize=18)
ylabels=[int(round(freq[i],0)) for i in range(0,Nf,4)]
plt.yticks(range(len(freq))[::4],ylabels)
#plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='white',alpha=0.5,linewidth=3)
plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.title('')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/PLVmean0_paper.png')
plt.close()



#%%% Plot mPLV in band

frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
phase=np.zeros((Nf,Nc,Nt))
for k in range(Nc):
    phase[:,k,:]=np.load(outputdir+'/phasesmultitaper_ch'+str(k)+'.npy')

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000

phase_epochs=np.zeros((Npics,Nf,Nc,Nt_trials))
for i in range(Npics):
    j=picstidx[i]
    phase_epochs[i]=phase[:,:,j-Npre:j+Npost] # phase=np.zeros((Nfreq,Nc,Nt))


PLVt=np.zeros((Nf,Nc,Nc,Nt_trials))
for (k1,k2) in tqdm.tqdm(contactpairs):
    PLVt[:,k1,k2,:]=np.abs((np.exp((phase_epochs[:,:,k1,:]-phase_epochs[:,:,k2,:])*1j)).mean(axis=0))    
mask_freq_global=(frequencies>=6)*(frequencies<=16)
PLVt=PLVt[mask_freq_global,:,:,:].mean(axis=0)
for (k1,k2) in tqdm.tqdm(contactpairs):
    PLVt[k2,k1,:]=PLVt[k1,k2,:]


PLV=np.zeros((len(K1s), Nt_trials))
for k in range(len(K1s)):
    (k1,k2)=contactpairs[k]
    PLV[k,:] = PLVt[k1,k2]


aux=PLV
NNN=aux.shape[0]
m=aux.mean(axis=0)
s=aux.std(axis=0) / np.sqrt(NNN)
msu=m+s
msl=m-s

plt.figure(figsize=(10,6)) 
plt.fill_between(T, msl, msu, color='k', alpha=0.2)
plt.plot(T,m,color='k')
#plt.hlines(hlines,T[0],T[-1],color='k',linestyle='dashed')        
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='k',alpha=0.2)
#plt.title(contacts[k])
#plt.yticks(hlines,[-50,50]*Npics+[-20,20])
plt.xlabel('')
plt.ylabel('')
#plt.title(voiname)
xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
plt.yticks(fontsize=18)


plt.savefig(outputdir+'/mPLV_band.png')



pre_idx=(T<-100).nonzero()[0][-1]
win0=(T>300).nonzero()[0][0]
win1=(T<600).nonzero()[0][-1]
#[(FCmean[i,win0:win1]<FCmean[i,:preidx]).sum(axis=-1) for i in range(Npics)]
#[(FCmean[i,win0:win1]<FCmean[i,:preidx]).sum(axis=-1)/(preidx) for i in range(Npics)]
D0=PLV[:,win0:win1].mean(axis=-1)
D1=PLV[:,:pre_idx].mean(axis=-1)
print(st.ranksums(D0,D1))

# # Epoch data
# FCmean=np.zeros((Npics, 2))
# for i in range(Npics):T
#     j=picstidx[i]
#     data_epoch=y[:,j-Npre:j+Npost]
#     fc=np.corrcoef(data_epoch[:,:int(fs/2)])
#     FCmean[i,0]=fc[K1s,K2s].mean()
#     fc=np.corrcoef(data_epoch[:,int(fs/2):3*int(fs/2)])
#     FCmean[i,1]=fc[K1s,K2s].mean()
    

# plt.plot(FCmean)


#%%% PLV strength per channel

PLV_strength=np.zeros((Nc, Nt_trials))   
for k in range(Nc):
    aux=PLVt[K1s,K2s,:]
    idx=[i for i in range(len(K1s)) if K1s[i]==k]+[i for i in range(len(K1s)) if K2s[i]==k]
    aux=aux[idx,:]
    PLV_strength[k,:]=np.mean(aux,axis=0)



m=PLV_strength
m=(m.T-m[:,:pre_idx].mean(axis=1)).T
plt.figure('PLV strength',figsize=(10,6))
plt.pcolormesh(T, range(Nc), m, cmap='jet', shading='none',vmin=-0.04,vmax=0.04)
xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
ylabels=[whereMVV[c] for c in contacts]
plt.yticks(range(Nc),ylabels,fontsize=9)
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.title('PLV strength')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/PLVstrength.png')
plt.close() 

#%%% PLV strength in pre and post

win0=(T>300).nonzero()[0][0]
win1=(T<600).nonzero()[0][-1]

plt.figure(figsize=(10,6)) 


m=np.zeros(Nc)
s=np.zeros(Nc)
for k in range(Nc):
    aux=PLVt[K1s,K2s,:]
    idx=[i for i in range(len(K1s)) if K1s[i]==k]+[i for i in range(len(K1s)) if K2s[i]==k]
    aux=aux[idx,:]
    PLVinperiod=aux[:,:pre_idx].mean(axis=-1)
    m[k]=PLVinperiod.mean(axis=0)
    s[k]=PLVinperiod.std(axis=0)/np.sqrt(Nc)
#m=mm.mean(axis=0)
#s=mm.std(axis=0)/np.sqrt(Nc)
plt.fill_between(np.arange(Nc), m-s, m+s, color='b', alpha=0.2)
plt.plot(np.arange(Nc),m,label='baseline')


m=np.zeros(Nc)
s=np.zeros(Nc)
for k in range(Nc):
    aux=PLVt[K1s,K2s,:]
    idx=[i for i in range(len(K1s)) if K1s[i]==k]+[i for i in range(len(K1s)) if K2s[i]==k]
    aux=aux[idx,:]
    PLVinperiod=aux[:,win0:win1].mean(axis=-1)
    m[k]=PLVinperiod.mean(axis=0)
    s[k]=PLVinperiod.std(axis=0)/np.sqrt(Nc)
#mm=FC_strength[:,win0:win1].mean(axis=-1)
#m=mm.mean(axis=0)
#s=mm.std(axis=0)/np.sqrt(Nc)
plt.fill_between(np.arange(Nc), m-s, m+s, color='r', alpha=0.2)
plt.plot(np.arange(Nc),m,label='300-600 ms')

plt.legend()
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(outputdir+'/PLV strenght pre post_.png')


#%%% PLV in pre and post plot

plt.figure(figsize=(8,6)) 

Tw=np.arange(Nw)*window_step/fs*1000-500
pre_idx=( (Tw+window_len/fs*1000) >0).nonzero()[0][0]
PLVinperiod0=PLVt[:,:,:pre_idx].mean(axis=-1)
win0=(Tw>300).nonzero()[0][0]
win1=(Tw<600).nonzero()[0][-1]
PLVinperiod1=PLVt[:,:,win0:win1].mean(axis=-1)
plt.imshow(PLVinperiod1-PLVinperiod0,vmin=-0.1,vmax=0.1,origin='lower')
plt.colorbar()
ylabels=[whereMVV[c] for c in contacts]
plt.yticks(range(Nc),ylabels,fontsize=7)
plt.xticks(range(Nc),ylabels,fontsize=7,rotation=90)
plt.xlabel('contact ROI')
plt.ylabel('contact ROI')
plt.savefig(outputdir+'/PLV post-pre.png')


#%% Plot 2 globals

# mPLV
mPLV=np.load(outputdir+'/mPLV.npy')
(Nf,Nt_trials)=mPLV.shape
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000
frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
mask_freq_global=(frequencies>=6)*(frequencies<=16)
norm_idx=(T<-100).nonzero()[0][-1]
#mPLV=((mPLV.T-mPLV[:,:norm_idx].mean(axis=-1))/mPLV[:,:norm_idx].std(axis=-1)).T
G1=mPLV[mask_freq_global,:].mean(axis=0)
m=G1.mean()
s=G1.std()
G1=(G1-m)/s

G2=np.load(outputdir+'/FCabs2mean_bandbroadZ.npy')
m=G2.mean()
s=G2.std()
G2=(G2-m)/s

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000

plt.figure('Globals',figsize=(6.7,3))
plt.plot(T,G1,label='mPLV')
plt.plot(T,G2, label='mFC')
plt.legend()
xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
plt.yticks(fontsize=0)
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.savefig(outputdir+'/2globals.png')
plt.close()

#%% LOCAL - GLOBAL

#%%% Contact activations

Local=np.zeros((Nc,Nf,Nt_trials))
for k in range(Nc):
    Local[k]=np.load(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'.npy')
mask_freq_local=(frequencies>=64)*(frequencies<=256)
var1=Local[:,mask_freq_local,:].mean(axis=1)

# Epoch data
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000
mask_t=(T<600)*(T>300)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('MA',figsize=(2,6))
plt.plot(MA[argsort_ind],np.arange(Nc))
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.ylim([-0.5,Nc-0.5])
plt.xticks([0.8,1,1.2])
ax=plt.gca()
ax.set_position([0.1,0.125,0.775,0.755])
plt.savefig(outputdir+'/MA.png')
plt.close()

m=var1
plt.figure('Contact activations',figsize=(8,6))
plt.pcolormesh(T, np.arange(Nc), m[argsort_ind,:], vmin=0.7, vmax=1.3, cmap='jet', shading='nearest')
xlabels=['' for i in np.arange(-500,1501,100)]
xlabels[0]='-500'
xlabels[5]='0'
xlabels[10]='500'
xlabels[15]='1000'
xlabels[20]='1500'
plt.xticks(np.arange(-500,1501,100),xlabels, fontsize=18)
ylabels=[whereMVV[c] for c in [contacts[i] for i in argsort_ind]]
plt.yticks(range(Nc),ylabels,fontsize=9)
plt.vlines([0,ISI*1000],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.title('')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/contactactivations.png')
plt.close()


#%%% 1 mFC corr Spearman

var2=np.load(outputdir+'/FCabs2mean_bandbroadZ.npy')

Local=np.zeros((Nc,Nf,Nt_trials))
for k in range(Nc):
    Local[k]=np.load(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'.npy')
mask_freq_local=(frequencies>=64)*(frequencies<=256)
var1=Local[:,mask_freq_local,:].mean(axis=1)

cors=np.zeros(Nc)
pvalues=np.zeros(Nc)
cors_part=np.zeros(Nc)
pvalues_part=np.zeros(Nc)

for k in range(Nc):
    print(k)
    
    (r,p)=spearmanr_circular_shifts(var1[k],var2,lag_min=400,lag_max=None)
    cors[k]=r
    pvalues[k]=p
    
    (r,p)=spearmanr_partial_circular_shifts(var1,k,var2,lag_min=400,lag_max=None)
    cors_part[k]=r
    pvalues_part[k]=p

np.save(outputdir+'/corr_margSpearman_mFC_rval.npy',cors)
np.save(outputdir+'/corr_margSpearman_mFC_pval.npy',pvalues)

np.save(outputdir+'/corr_partSpearman_mFC_rval.npy',cors_part)
np.save(outputdir+'/corr_partSpearman_mFC_pval.npy',pvalues_part)


#%%% 1.1 Plot mFC corr_margSpearman

cors=np.load(outputdir+'/corr_margSpearman_mFC_rval.npy')
pvalues=np.load(outputdir+'/corr_margSpearman_mFC_pval.npy')
pval=pvalues<0.001  
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000
mask_t=(T<600)*(T>300)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_margSpearman_mFC.png')
plt.close()

#%%% 1.2 Plot mFC corr_semipartSpearman

cors=np.load(outputdir+'/corr_partSpearman_mFC_rval.npy')
pvalues=np.load(outputdir+'/corr_partSpearman_mFC_pval.npy')
pval=pvalues<0.001  
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000
mask_t=(T<600)*(T>300)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_partSpearman_mFC.png')
plt.close()


#%%% 1.3 mFC - local V1/V2 scatter plot

# # Epoch data
# Nt_trials=int((Pre+Post)*fs)
# Npre=int(Pre*fs)
# Npost=int(Post*fs)
# T=np.arange(-Npre,Npost)/fs*1000
# mask_t=(T<600)*(T>300)

# m=var1
# MA=var1[:,mask_t].mean(axis=-1)
# argsort_ind=MA.argsort()

# k=-2

# plt.figure(figsize=(10,6)) 
# plt.scatter(var1[argsort_ind][k][::100],var2[::100])
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.savefig(outputdir+'/scatter_V1_mFC.png')


#%%% 2 mPLV Spearman

Global=np.load(outputdir+'/mPLV.npy')
(Nf,Nt_trials)=Global.shape
mask_freq_global=(frequencies>=6)*(frequencies<=16)
var2=Global[mask_freq_global,:].mean(axis=0)

Local=np.zeros((Nc,Nf,Nt_trials))
for k in range(Nc):
    Local[k]=np.load(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'.npy')
mask_freq_local=(frequencies>=64)*(frequencies<=256)
var1=Local[:,mask_freq_local,:].mean(axis=1)

cors=np.zeros(Nc)
pvalues=np.zeros(Nc)
cors_part=np.zeros(Nc)
pvalues_part=np.zeros(Nc)

for k in range(Nc):
    print(k)
    
    (r,p)=spearmanr_circular_shifts(var1[k],var2,lag_min=400,lag_max=None)
    cors[k]=r
    pvalues[k]=p
    
    (r,p)=spearmanr_partial_circular_shifts(var1,k,var2,lag_min=400,lag_max=None)
    cors_part[k]=r
    pvalues_part[k]=p
             
np.save(outputdir+'/corr_margSpearman_mPLV_rval.npy',cors)
np.save(outputdir+'/corr_margSpearman_mPLV_pval.npy',pvalues)

np.save(outputdir+'/corr_partSpearman_mPLV_rval.npy',cors_part)
np.save(outputdir+'/corr_partSpearman_mPLV_pval.npy',pvalues_part)


#%%% 2.1 Plot mPLV corr_margSpearman

cors=np.load(outputdir+'/corr_margSpearman_mPLV_rval.npy')
pvalues=np.load(outputdir+'/corr_margSpearman_mPLV_pval.npy')
pval=pvalues<0.001  
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000
mask_t=(T<600)*(T>300)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_margSpearman_mPLV.png')
plt.close()

#%%% 2.2 Plot mPLV corr_semipartSpearman

cors=np.load(outputdir+'/corr_partSpearman_mPLV_rval.npy')
pvalues=np.load(outputdir+'/corr_partSpearman_mPLV_pval.npy')
pval=pvalues<0.001
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials=int((Pre+Post)*fs)
Npre=int(Pre*fs)
Npost=int(Post*fs)
T=np.arange(-Npre,Npost)/fs*1000
mask_t=(T<600)*(T>300)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_partSpearman_mPLV.png')
plt.close()

#%%% 2.3 mPLV - local V1/V2 scatter plot

# # Epoch data
# Nt_trials=int((Pre+Post)*fs)
# Npre=int(Pre*fs)
# Npost=int(Post*fs)
# T=np.arange(-Npre,Npost)/fs*1000
# mask_t=(T<600)*(T>300)

# m=var1
# MA=var1[:,mask_t].mean(axis=-1)
# argsort_ind=MA.argsort()

# k=-2

# plt.figure(figsize=(10,6)) 
# plt.scatter(var1[argsort_ind][k][::100],var2[::100])
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.savefig(outputdir+'/scatter_V1_mPLV.png')   
        

       



#%% Plot 2 globals REC

# mPLV
mPLV=np.load(outputdir+'/mPLV1.npy')
(Nf,Nt_trials_ation)=mPLV.shape
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
frequencies=2**np.arange(1,np.log2(512+1),1/4)
Nf=frequencies.shape[0]
mask_freq_global=(frequencies>=6)*(frequencies<=16)
#norm_idx=(T<-100).nonzero()[0][-1]
#mPLV=((mPLV.T-mPLV[:,:norm_idx].mean(axis=-1))/mPLV[:,:norm_idx].std(axis=-1)).T
G1=mPLV[mask_freq_global,:].mean(axis=0)
m=G1.mean()
s=G1.std()
G1=(G1-m)/s


G2=np.load(outputdir+'/FCabs2mean_1_bandbroadZ.npy')
m=G2.mean()
s=G2.std()
G2=(G2-m)/s

# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
#mask_t=(T_action<200)*(T_action>-100)

plt.figure('Globals',figsize=(6.7,3))
plt.plot(T_action,G1,label='mPLV')
plt.plot(T_action,G2, label='mFC')
plt.legend()
xlabels=['' for i in np.arange(-1000,501,100)]
#xlabels[1]='-400'
#xlabels[3]='-200'
#xlabels[5]=''
xlabels[6]='-400'
xlabels[8]='-200'
xlabels[10]='0'
xlabels[12]='200'
xlabels[14]='400'
plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
plt.yticks(fontsize=0)
plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
plt.savefig(outputdir+'/2globals1.png')
plt.close()



#%% LOCAL - GLOBAL REC

#%%% Contact activation REC

# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000

Local=np.zeros((Nc,Nf,Nt_trials_ation))
for k in range(Nc):
    Local[k]=np.load(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'_rec.npy')
mask_freq_local=(frequencies>=64)*(frequencies<=256)
var1=Local[:,mask_freq_local,:].mean(axis=1)


mask_t=(T_action<200)*(T_action>-100)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('MA',figsize=(2,6))
plt.plot(MA[argsort_ind],np.arange(Nc))
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.ylim([-0.5,Nc-0.5])
plt.xticks([0.9,1,1.1])
ax=plt.gca()
ax.set_position([0.1,0.125,0.775,0.755])
plt.savefig(outputdir+'/MA1.png')
plt.close()

m=var1
plt.figure('Contact activations',figsize=(8,6))
plt.pcolormesh(T_action, np.arange(Nc), m[argsort_ind,:], vmin=0.7, vmax=1.3, cmap='jet', shading='nearest')
xlabels=['' for i in np.arange(-1000,501,100)]
#xlabels[1]='-400'
#xlabels[3]='-200'
#xlabels[5]=''
xlabels[6]='-400'
xlabels[8]='-200'
xlabels[10]='0'
xlabels[12]='200'
xlabels[14]='400'
plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
ylabels=[whereMVV[c] for c in [contacts[i] for i in argsort_ind]]
plt.yticks(range(Nc),ylabels,fontsize=9)
plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/contactactivations1.png')
plt.close()


#%%% OLD: Contact activations NONREC

def get_frequencies():
    return 2**np.arange(1,np.log2(512+1),1/4)
freq=get_frequencies()
Nf=freq.shape[0]
# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)

Local=np.zeros((Nc,Nf,Nt_trials_action))
for k in range(Nc):
    Local[k]=np.load(outputdir+'/spectrogram_median_ch'+str(k)+'_nonrec.npy')
mask_freq_local=(frequencies>=64)*(frequencies<=256)
var1=Local[:,mask_freq_local,:].mean(axis=1)
m=var1
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()
plt.figure('Contact activations',figsize=(8,6))
plt.pcolormesh(T_action, np.arange(Nc), m[argsort_ind,:], vmin=0.7, vmax=1.3, cmap='jet', shading='nearest')
xlabels=['' for i in np.arange(-1000,501,100)]
#xlabels[1]='-400'
#xlabels[3]='-200'
#xlabels[5]=''
xlabels[6]='-400'
xlabels[8]='-200'
xlabels[10]='0'
xlabels[12]='200'
xlabels[14]='400'
plt.xticks(np.arange(-1000,501,100),xlabels, fontsize=18)
ylabels=[whereMVV[c] for c in [contacts[i] for i in argsort_ind]]
plt.yticks(range(Nc),ylabels,fontsize=9)
plt.vlines([0],plt.ylim()[0],plt.ylim()[1],color='r',alpha=0.5)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=18)
plt.savefig(outputdir+'/contactactivations0.png')
plt.close()


# Plot MA aligned to button press NONREC

plt.figure('corr',figsize=(2,6))
plt.plot(MA[argsort_ind],np.arange(Nc))
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.ylim([-0.5,Nc-0.5])
plt.xticks([0.9,1,1.1])
ax=plt.gca()
ax.set_position([0.1,0.125,0.775,0.755])
plt.savefig(outputdir+'/MA0.png')
plt.close()



#%%% 1 mFC corr Spearman REC

var2=np.load(outputdir+'/FCabs2mean_1_bandbroadZ.npy')

# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)

Local=np.zeros((Nc,Nf,Nt_trials_action))
for k in range(Nc):
    Local[k]=np.load(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'_rec.npy')
mask_freq_local=(frequencies>=64)*(frequencies<=256)
var1=Local[:,mask_freq_local,:].mean(axis=1)

cors=np.zeros(Nc)
pvalues=np.zeros(Nc)
cors_part=np.zeros(Nc)
pvalues_part=np.zeros(Nc)

for k in range(Nc):
    print(k)
    
    (r,p)=spearmanr_circular_shifts(var1[k,-fs:-200],var2[-fs+200:],lag_min=400,lag_max=None)
    cors[k]=r
    pvalues[k]=p
    
    (r,p)=spearmanr_partial_circular_shifts(var1[:,-fs:-200],k,var2[-fs+200:],lag_min=400,lag_max=None)
    cors_part[k]=r
    pvalues_part[k]=p

np.save(outputdir+'/corr_margSpearman_mFC1_rval.npy',cors)
np.save(outputdir+'/corr_margSpearman_mFC1_pval.npy',pvalues)

np.save(outputdir+'/corr_partSpearman_mFC1_rval.npy',cors_part)
np.save(outputdir+'/corr_partSpearman_mFC1_pval.npy',pvalues_part)


#%%% 1.1 Plot mFC corr_margSpearman

cors=np.load(outputdir+'/corr_margSpearman_mFC1_rval.npy')
pvalues=np.load(outputdir+'/corr_margSpearman_mFC1_pval.npy')
pval=pvalues<0.001
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_margSpearman_mFC1.png')
plt.close()


#%%% 1.2 Plot mFC corr_semipartSpearman

cors=np.load(outputdir+'/corr_partSpearman_mFC1_rval.npy')
pvalues=np.load(outputdir+'/corr_partSpearman_mFC1_pval.npy')
pval=pvalues<0.001  
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_partSpearman_mFC1.png')
plt.close()



#%%% 1.3 mFC - local M1 scatter plot

# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)

MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()


k=-3

par=200

plt.figure(figsize=(10,6)) 
plt.scatter(var1[argsort_ind][k][-fs:-par:100],var2[-fs+par::100])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(outputdir+'/scatter M1 mFC_.png')

#%%% 1.3 mFC - local STG scatter plot

# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)

MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()


k=-5

plt.figure(figsize=(10,6)) 
plt.scatter(var1[argsort_ind][k][-fs:-par:100],var2[-fs+par::100])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(outputdir+'/scatter STG mFC_.png')




#%%% 2 mPLV Spearman REC

Global=np.load(outputdir+'/mPLV1.npy')
(Nf,Nt_trials)=Global.shape
mask_freq_global=(frequencies>=6)*(frequencies<=16)
var2=Global[mask_freq_global,:].mean(axis=0)

# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)

Local=np.zeros((Nc,Nf,Nt_trials_action))
for k in range(Nc):
    Local[k]=np.load(outputdir+'/spectrogram_multitaper_median_ch'+str(k)+'_rec.npy')
mask_freq_local=(frequencies>=64)*(frequencies<=256)
var1=Local[:,mask_freq_local,:].mean(axis=1)

cors=np.zeros(Nc)
pvalues=np.zeros(Nc)
cors_part=np.zeros(Nc)
pvalues_part=np.zeros(Nc)

for k in range(Nc):
    print(k)
    
    (r,p)=spearmanr_circular_shifts(var1[k,-fs:],var2[-fs:],lag_min=400,lag_max=None)
    cors[k]=r
    pvalues[k]=p
    
    (r,p)=spearmanr_partial_circular_shifts(var1,k,var2,lag_min=400,lag_max=None)
    cors_part[k]=r
    pvalues_part[k]=p
             
np.save(outputdir+'/corr_margSpearman_mPLV1_rval.npy',cors)
np.save(outputdir+'/corr_margSpearman_mPLV1_pval.npy',pvalues)

np.save(outputdir+'/corr_partSpearman_mPLV1_rval.npy',cors_part)
np.save(outputdir+'/corr_partSpearman_mPLV1_pval.npy',pvalues_part)




#%%% 2.1 Plot mPLV corr_margSpearman

cors=np.load(outputdir+'/corr_margSpearman_mPLV1_rval.npy')
pvalues=np.load(outputdir+'/corr_margSpearman_mPLV1_pval.npy')
pval=pvalues<0.001
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_margSpearman_mPLV1.png')
plt.close()


#%%% 2.2 Plot mPLV corr_semipartSpearman

cors=np.load(outputdir+'/corr_partSpearman_mPLV1_rval.npy')
pvalues=np.load(outputdir+'/corr_partSpearman_mPLV1_pval.npy')
pval=pvalues<0.001  
cors[np.abs(cors)<0.3]=0

# MA
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)
MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()

plt.figure('corr',figsize=(3,6))
idx=pval[argsort_ind].nonzero()[0]
plt.barh(np.arange(Nc),cors[argsort_ind])
plt.barh(np.arange(Nc)[idx],cors[argsort_ind][idx],color='r')
plt.yticks(np.arange(Nc),[whereMVV[c] for c in [contacts[i] for i in argsort_ind]],fontsize=9)
plt.xticks([-0.5,0,0.5],fontsize=18)
plt.ylim([-0.5,Nc-0.5])
ax=plt.gca()
ax.set_position([0.2,0.125,0.775,0.755])

plt.savefig(outputdir+'/corr_partSpearman_mPLV1.png')
plt.close()

#%%% 2.3 mPLV - local M1 scatter plot

# Epoch data
Nt_trials_action=int((Pre+2*Pre)*fs)
Npre=int(Pre*fs)
Npost=int((ISI+Pre)*fs)
T_action=np.arange(-2*Npre,Npre)/fs*1000
mask_t=(T_action<200)*(T_action>-100)

MA=var1[:,mask_t].mean(axis=-1)
argsort_ind=MA.argsort()


k=-3

par=200

plt.figure(figsize=(10,6)) 
plt.scatter(var1[argsort_ind][k][-fs:-par:100],var2[-fs+par::100])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(outputdir+'/scatter M1 mPLV_.png')


