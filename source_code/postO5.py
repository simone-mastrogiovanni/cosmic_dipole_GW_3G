import numpy as np
import pickle
import icarogw
import os
import bilby
import tqdm
import copy
import copy as cp

def calc_ini_prior_mdet(m1,m2,mode):
    '''
    Given m1 and m2 in detector frame, it will give you back the prior used for the simulation (detector frame)

    Parameters
    ----------
    m1,m2: array of detector frame masses, solar masses
    mode: either 'BNS' or 'BBH'
    '''
    
    ini_prob=np.zeros_like(m1)
    if mode=='BNS':
        pm1=bilby.core.prior.PowerLaw(alpha=0.,minimum=1.,maximum=5.,name='mass_1')
        pm2=bilby.core.prior.PowerLaw(alpha=0.,minimum=1.,maximum=5.,name='mass_2')
    elif mode=='BBH':
        pm1=bilby.core.prior.PowerLaw(alpha=-2.0,minimum=2.,maximum=500,name='mass_1')
        pm2=bilby.core.prior.PowerLaw(alpha=1.0,minimum=2.,maximum=500,name='mass_2')
        
    for i in range(len(m1)):
        pm2.maximum=m1[i]
        ini_prob[i]=pm1.prob(m1[i])*pm2.prob(m2[i])
    return ini_prob


def create_tab(file,mode):
    '''
    Given a file of detections from the 'detection_files' folder, it will create a dictionary of detected events

    Parameters
    ----------
    file: path to the file
    mode: either 'BNS' or 'BBH'
    '''
    
    data=pickle.load(open(file,'rb'))
    pars=list(data[0]['intrinsic'].keys())+list(data[0]['extrinsic'].keys())+\
    ['optimal_SNR2_IFO_'+ifo for ifo in data[0]['ifo_on']]+['matched_filter_SNR2_IFO_'+ifo for ifo in data[0]['ifo_on']]
    event_dict={var:np.zeros(len(data)) for var in pars}
    for i in range(len(data)):
        for var in pars:
            if var in list(data[0]['intrinsic'].keys()):
                event_dict[var][i]=data[i]['intrinsic'][var]
            elif var in list(data[0]['extrinsic'].keys()):
                event_dict[var][i]=data[i]['extrinsic'][var]
            else:
                event_dict[var][i]=data[i][var]
    event_dict['prob_masses']=calc_ini_prior_mdet(event_dict['mass1'],event_dict['mass2'],mode)
    return event_dict

def calc_det_rate_per_year(event_dict,ifos,duties,snr_thr,mass_model,rate_model,R0):
    '''
    The function will caclulate the number of expected detections per year
    
    Parameters
    ----------
    event_dict: Dictionary of detected events from create_tab
    ifos: list containing ifos, e.g. ['H1','L1','V1']
    duties: list of duty cycles, e.g. [0.7,0.8,1.0]
    snr_thr: Optimal Network snr threshold for detection
    mass_model: icarogw mass class
    rate_model: icarogw rate class
    R0: merger rate today per Gpc-3 yr-1
    '''
    
    pars=list(event_dict.keys())
    tot_events=len(event_dict[pars[0]])
    totrate=[]
    event_dict['redshift_lim']=np.zeros(tot_events) 
    event_dict['Ndetyr']=np.zeros(tot_events) 
    event_dict['ifo_on']=[]
    for i in tqdm.tqdm(range(tot_events)):
        snr_net=[]
        ifo_on=[]
        for j,ifo in enumerate(ifos):
            if np.random.rand()<=duties[j]:
                snr_net.append(event_dict['optimal_SNR2_IFO_'+ifo][i])
                ifo_on.append(ifo)
        event_dict['ifo_on'].append(cp.copy(ifo_on))
        if len(snr_net)==0:
            event_dict['redshift_lim'][i]=0.
            event_dict['Ndetyr'][i]=0.
        snr_net=np.sqrt(np.sum(snr_net))
        dldet=snr_net/snr_thr
        zdet=rate_model.cosmo.z_at_dl(dldet)[0]
        event_dict['redshift_lim'][i]=zdet
        zarray=np.linspace(0,zdet,1000)
        m1s,m2s=event_dict['mass1'][i]/(1+zarray),event_dict['mass2'][i]/(1+zarray)
        toint=R0*rate_model.prob_astro(zarray)*mass_model.joint_prob(m1s,m2s)/(event_dict['prob_masses'][i]*np.power(1+zarray,2.))        
        event_dict['Ndetyr'][i]=np.trapz(toint,zarray)
    event_dict['ifo_on']=np.array(event_dict['ifo_on'],dtype=object)
    return np.sum(event_dict['Ndetyr'])/tot_events


def resample(event_dict,mass_model,rate_model,Nsamp=5000):
    '''
    The function will caclulate the number of expected detections per year
    
    Parameters
    ----------
    event_dict: Dictionary of detected events processed by calc_det_rate_per_year
    mass_model: icarogw mass class
    rate_model: icarogw rate class
    Nsamp: How many events you want
    '''
    
    chx=np.random.choice(len(event_dict['Ndetyr']),replace=True,
                          size=Nsamp,p=event_dict['Ndetyr']/event_dict['Ndetyr'].sum())
    
    zarr=np.zeros(Nsamp)
    m1arr=np.zeros(Nsamp)
    m2arr=np.zeros(Nsamp)
    snr_array=np.zeros(Nsamp)
    
    for i,cl in tqdm.tqdm(enumerate(chx)):
        zlim=event_dict['redshift_lim'][cl]
        zarray=np.linspace(0,zlim,1000)
        m1s,m2s=event_dict['mass1'][cl]/(1+zarray),event_dict['mass2'][cl]/(1+zarray)
        toint=rate_model.prob_astro(zarray)*mass_model.joint_prob(m1s,m2s)/(event_dict['prob_masses'][cl]*np.power(1+zarray,2.))
        indxsel=np.random.choice(len(toint),size=1,p=toint/toint.sum())[0]

        zarr[i]=zarray[indxsel]
        m1arr[i]=m1s[indxsel]
        m2arr[i]=m2s[indxsel]
        snr_array[i]=rate_model.cosmo.dl_at_z(zlim)/rate_model.cosmo.dl_at_z(zarr[i])
        
        
    out_dict={var:event_dict[var][chx] for var in event_dict.keys()}
    out_dict['redshift']=zarr
    out_dict['dl']=rate_model.cosmo.dl_at_z(zarr)
    out_dict['Mc']=np.power(m1arr*m2arr,3./5)*np.power(m1arr+m2arr,-1./5.)*(1+zarr)
    out_dict['mass1_source']=m1arr
    out_dict['mass2_source']=m2arr
    out_dict['snr_over_snr_thr']=snr_array
    
    return out_dict
    
    
