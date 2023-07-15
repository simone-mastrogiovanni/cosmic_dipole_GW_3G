import sys
sys.path.append('./source_code')

import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import bilby
import icarogw
from astropy import constants
from healpy.newvisufunc import projview, newprojplot
from tqdm import tqdm
from scipy.stats import ncx2
import pickle
import copy as cp


def radec2indeces(ra,dec,nside):
    '''
    Converts RA and DEC to healpy indeces
    
    Parameters
    ----------
    ra, dec: xp.array
        arrays with RA and DEC in radians
    nside: int
        nside for healpy
    
    Returns
    -------
    healpy indeces as numpy array
    '''
    theta = np.pi/2.0 - dec
    phi = ra
    return hp.ang2pix(nside, theta, phi)

def fit_PSD_factor(list_of_ASD,fmin,fmax):
    '''
    Provided a list of txt files with detectors ASDs, it will return the interpolant of the network sensitivity
    for the 0 PN expansion model of the signal
    
    Parameters
    ----------
    list_of_ASD: list
        List of string containing the file names, files should have frequency on first column and ASD in Hz^-0.5 on second
    fmin, fmax: float
        minimum and maximum for the interpolation, outside the range the sensitivity will be zero.
    '''
    
    interpolant=[]
    for ff in list_of_ASD:
        data=np.genfromtxt(ff)
        cum_PSD=cumtrapz(np.power(data[:,0],-7./3)/(data[:,1]**2.),data[:,0])
        interpolant.append(interp1d(data[:-1,0],cum_PSD,bounds_error=False,fill_value=(0.,cum_PSD[-1])))    
    freq_eval=np.logspace(np.log10(fmin),np.log10(fmax),10000)
    evals=np.zeros_like(freq_eval)
    for i in range(len(interpolant)):
        evals+=interpolant[i](freq_eval)
    return interp1d(freq_eval,evals,bounds_error=False,fill_value=(0.,evals[-1]))


def draw_binaries(zprior,mass_prior,Nsample):
    '''
    Draw the redshift, source masses, cosiota and sky position of the GW events. Then it computes detector frame masses
    and luminosity distances.
    
    Parameters
    ----------
    zprior,mass_prior:
        icarogw redsfhit and mass prior
    Nsample: int
        Number of binaries to draw.
    
    Returns
    -------
    Dictionary containing the binary parameters
    
    '''
    binaries_dict={}
    binaries_dict['z']=zprior.sample(Nsample)
    binaries_dict['dl']=zprior.cosmo.dl_at_z(binaries_dict['z']) # Luminosity distance in Mpc.
    binaries_dict['mass1_source'],binaries_dict['mass2_source']=mass_prior.sample(Nsample) # Solar masses
    binaries_dict['mass1'],binaries_dict['mass2']=binaries_dict['mass1_source']*(1+binaries_dict['z']),binaries_dict['mass2_source']*(1+binaries_dict['z'])
    binaries_dict['cosiota']=np.random.uniform(low=-1,high=1,size=Nsample)
    binaries_dict['phi'],binaries_dict['theta']=np.random.uniform(low=0.,high=2*np.pi,size=Nsample),np.arccos(np.random.uniform(low=-1,high=1,size=Nsample))
    return binaries_dict

class double_positive_gaussian(object):
    '''
    A small class to sample from a Double gaussian with m2<m1
    '''
    def __init__(self,mu_g,sigma_g):
        self.mu_g=mu_g
        self.sigma_g=sigma_g
    
    def sample(self,Nsamp):
        gen=bilby.core.prior.TruncatedGaussian(self.mu_g,self.sigma_g,minimum=0,maximum=np.inf)
        mass1,mass2=gen.sample(Nsamp),gen.sample(Nsamp)
        swapper=np.where(mass1<mass2)[0]
        mass1[swapper],mass2[swapper]=mass2[swapper],mass1[swapper]
        return mass1,mass2

def caluclate_detected_binaries(binaries_dict,interp,num_detectors,dipole_magnitude,snr_thr,binary_type):
    '''
    Calculates the detected binaries. It also applies the corrections to the luminosity distance and sky position
    given by the cosmic dipole
    
    Parameters
    ----------
    binaries_dict: dict
        Dictionary of binaries
    interp: interpolant
        Interpolant of the ASD to compute the SNR from its function
    num_detectors: int
        Number of detectors in the network
    dipole_magnitude: float
        Dipole magnitude in v/c
    snr_thr: float
        SNR threshold for the detection
    binary_type: string
        Either BNS or BBH.
    
    Returns
    -------
    Dictionary of detected binaries.
    '''
    
    
    
    dipole_projection=np.cos(binaries_dict['theta'])
    binaries_dict['aberration']=-dipole_magnitude*np.sin(binaries_dict['theta'])
    binaries_dict['phi_original'],binaries_dict['theta_original']=cp.copy(binaries_dict['phi']),cp.copy(binaries_dict['theta'])
    
    # Add noise scatter to aberration
    binaries_dict['aberration']+=(np.random.uniform(low=-3,high=3,size=len(binaries_dict['aberration']))*np.pi/180.)
    
    #Abberrated
    binaries_dict['theta']=binaries_dict['theta']+binaries_dict['aberration']
        
    idx=np.where((binaries_dict['theta']>np.pi) & (binaries_dict['phi']>np.pi))[0]
    binaries_dict['phi'][idx],binaries_dict['theta'][idx]= binaries_dict['phi'][idx]-np.pi,binaries_dict['theta'][idx]-binaries_dict['aberration'][idx]

    idx=np.where((binaries_dict['theta']>np.pi) & (binaries_dict['phi']<np.pi))[0]
    binaries_dict['phi'][idx],binaries_dict['theta'][idx]= binaries_dict['phi'][idx]+np.pi,binaries_dict['theta'][idx]-binaries_dict['aberration'][idx]
    
    idx=np.where((binaries_dict['theta']<0.) & (binaries_dict['phi']>np.pi))[0]
    binaries_dict['phi'][idx],binaries_dict['theta'][idx]= binaries_dict['phi'][idx]-np.pi,binaries_dict['theta'][idx]-binaries_dict['aberration'][idx]

    idx=np.where((binaries_dict['theta']<0.) & (binaries_dict['phi']<np.pi))[0]
    binaries_dict['phi'][idx],binaries_dict['theta'][idx]= binaries_dict['phi'][idx]+np.pi,binaries_dict['theta'][idx]-binaries_dict['aberration'][idx]
    
    
    # Modifies masses and distance according to dipole
    binaries_dict['mass1'], binaries_dict['mass2'] = binaries_dict['mass1']*(1-dipole_magnitude*dipole_projection), binaries_dict['mass2']*(1-dipole_magnitude*dipole_projection)
    binaries_dict['dl']*=(1-dipole_magnitude*dipole_projection)
    
    chirp_mass=np.power(binaries_dict['mass1']*binaries_dict['mass2'],3/5.)/np.power(binaries_dict['mass1']+binaries_dict['mass2'],1/5.)
    chirp_mass*=1.99e30
    #chirp_mass*=(1-dipole_magnitude*dipole_projection)
    
    Mtot=binaries_dict['mass1']+binaries_dict['mass2']
    Mtot*=4.9e-6
    #Mtot*=(1-dipole_magnitude*dipole_projection)
    lso=1./(Mtot*np.pi*6**1.5)
    
    distance=binaries_dict['dl']*3.08e22
    #distance*=(1-dipole_magnitude*dipole_projection)
    Q=0.2*(np.power(0.5*(1+binaries_dict['cosiota']**2.),2.)+binaries_dict['cosiota']**2)
    
    binaries_dict['snr']=(5*np.power(constants.G.value*chirp_mass,5./3)*Q)/(6*np.power(constants.c.value,3.)*np.power(np.pi,4/3.)*np.power(distance,2.))
    if binary_type=='BNS':
        binaries_dict['snr']*=interp(lso)
    elif binary_type=='BBH':
        binaries_dict['snr']*=interp(2*lso)
        
    binaries_dict['snr']=np.sqrt(ncx2.rvs(num_detectors,binaries_dict['snr']))
    idx_det=np.where(binaries_dict['snr']>=snr_thr)[0]
    
    binaries_detected={key:binaries_dict[key][idx_det] for key in binaries_dict.keys()}
    
    return binaries_detected

def loop_detections(zp,mp,Nsamp,Ndet,interp,num_detectors,dipole_magnitude,snr_thr,binary_type):
    '''
    Just an helper function that will generate Nsamp injections in each loop until Ndet injections are detected
    
    Parameters
    ----------
    zp,mp: icarogw redshift and mass classes
    Nsamp: int
        Number of injections per loop
    Ndet: int
        Number of detection you want
        
    For the rest, see function above
    
    Returns
    -------
    Dictionary of detected binaries and number of injection Generated (even the not detected ones)
    
    '''
    
    bin_dict=draw_binaries(zp,mp,Nsamp)
    bin_det=caluclate_detected_binaries(bin_dict,interp,num_detectors,dipole_magnitude,snr_thr,binary_type)
    
    pbar=tqdm(total=Ndet,desc='Detected binaries')
    Ngen=0
    while len(bin_det['snr'])<Ndet:
        Ngen+=Nsamp
        bin_dict=draw_binaries(zp,mp,Nsamp)
        bin_det_sup=caluclate_detected_binaries(bin_dict,interp,num_detectors,dipole_magnitude,snr_thr,binary_type)
        bin_det={key:np.hstack([bin_det[key],bin_det_sup[key]]) for key in bin_det_sup}
        pbar.update(len(bin_det_sup['snr']))
    return bin_det,Ngen

def find_quantity_histo(phi,theta,nside,var=None):
    '''
    Helper function to find the histogram of a sky distribution
    
    Parmeters
    ---------
    phi,theta: rad
        Angles in spherical coordinates. Phi in 0, 2pi, theta in 0 pi.
    nside: int
        Nside for the Healpy pixelization
    var: array
        If passed, the function will compute the mean of the variable in each sky direction
    
    Returns
    -------
    Healpy Histogram map
    '''
    npixels=hp.nside2npix(nside)
    dOmega_sterad=4*np.pi/npixels
    dOmega_deg2=np.power(180/np.pi,2.)*dOmega_sterad
    indices=hp.ang2pix(nside, theta, phi)    
    mth_map=np.zeros(npixels)*np.nan
    for indx in range(npixels):
        ind=np.where(indices==indx)[0]
        if ind.size==0:
            continue
            
        if var is None:
            mth_map[indx] = len(ind)
        else:
            mth_map[indx] = np.mean(var[ind]) # Mean in the pixel
    # This is either a np or cp array
    mth_map[np.isnan(mth_map)]=hp.UNSEEN
    return hp.ma(mth_map)

    
def indices2radec(indices,nside):
    '''
    Converts healpy indeces to RA DEC
    
    Parameters
    ----------
    indices: xp.array
    nside: int
        nside for healpy
    
    Returns
    -------
    ra, dec: np.array
        arrays with RA and DEC in radians
    '''
    
    theta,phi= hp.pix2ang(nside,indices)
    return phi, np.pi/2.0-theta


def vc_estimator_map(phi,theta,nsidegw,nsidedip,hh=None,shuffle=False,var=None):
    '''
    Calculates sky map the dipole estimator given a list of GW detections.
    
    Parameters
    ----------
    phi,theta: rad
        Angles on the sphere of the GW detections
    nsidegw: int
        Healpy resolution used to bin the GW detections in the sky.
    nsidedip: 
        Healpy resolution to build the estimator map
    hh: healpy map
        Histogram of the GW detections in the sky, this will save some computing time
    Shuffle: bool
        If true, shuffle the GW detections sky position. Used to remove the effect of dipole.
    var: array
        If passed, the function will compute the mean of the variable in each sky direction
    
    Returns
    -------
    Map of the estimator and angles and also the histogram of the GW detections.
    
    '''
    npixels=hp.nside2npix(nsidedip)
    if hh is None:
        hh=find_quantity_histo(phi,theta,nsidegw,var=var)
    if shuffle:
        hh=np.random.permutation(hh)
        
    theta_pix,phi_pix = hp.pix2ang(nsidegw,np.arange(0,len(hh),1))
    n_gw=np.array([np.sin(theta_pix)*np.cos(phi_pix),np.sin(theta_pix)*np.sin(phi_pix),np.cos(theta_pix)])
    vc_map=np.zeros(npixels)
    for i in range(npixels):
        theta_dipole,phi_dipole= hp.pix2ang(nsidedip,i)
        n_dipole=np.array([np.sin(theta_dipole)*np.cos(phi_dipole),np.sin(theta_dipole)*np.sin(phi_dipole),np.cos(theta_dipole)])
        dipole_projection=np.dot(n_dipole,n_gw)
        
        if var is None:
            vc_map[i]=np.sum(hh*dipole_projection*1.5)/np.sum(hh)
        else:
            vc_map[i]=3*np.mean(hh*dipole_projection) # Mean over sky patches
        
    theta_map,phi_map=hp.pix2ang(nsidedip,np.arange(0,npixels,1))
    return vc_map,theta_map,phi_map,hh

def vc_estimator(phi,theta,phi_dipole,theta_dipole,nsidegw,hh=None,shuffle=False,var=None):
    '''
    Calculates the dipole estimator given a list of GW detections.
    
    Parameters
    ----------
    phi,theta: rad
        Angles on the sphere of the GW detections
    nsidegw: int
        Healpy resolution used to bin the GW detections in the sky.
    hh: healpy map
        Histogram of the GW detections in the sky, this will save some computing time
    Shuffle: bool
        If true, shuffle the GW detections sky position. Used to remove the effect of dipole.
     var: array
        If passed, the function will compute the mean of the variable in each sky direction
    
    
    Returns
    -------
    Estimator and histogram of GW detections
    '''
    
    if hh is None:
        hh=find_quantity_histo(phi,theta,nsidegw,var=var)
    if shuffle:
        hh=np.random.permutation(hh)
    
    theta_pix,phi_pix = hp.pix2ang(nsidegw,np.arange(0,len(hh),1))
    n_gw=np.array([np.sin(theta_pix)*np.cos(phi_pix),np.sin(theta_pix)*np.sin(phi_pix),np.cos(theta_pix)])
    n_dipole=np.array([np.sin(theta_dipole)*np.cos(phi_dipole),np.sin(theta_dipole)*np.sin(phi_dipole),np.cos(theta_dipole)])
    dipole_projection=np.dot(n_dipole.T,n_gw)
    
    if var is None:
        out= np.sum(hh*dipole_projection*1.5)/np.sum(hh)
    else:
        out= 3*np.mean(hh*dipole_projection)
    
    return out,hh

class hierarchical_likelihood(bilby.Likelihood):
    '''
    Bilby hierarchical likelihood model used to sample for the Dipole
    
    Parameters
    ----------
    phi,theta: rad
        Angles of the GW detections
    nsidegw: int
        Healpy resolution to bin the GW detections in the sky
    nsidedip: int
        Healpy resolution for the estimator (resolution for scalar products)
    shuffle: bool
        True if you want to remove the dipole by shuffling the sky.
    
    '''
    
    def __init__(self,phi,theta,nsidegw,nsidedip,shuffle=False):
        
        self.phi,self.theta,self.nsidegw,self.nsidedip=phi,theta,nsidegw,nsidedip
        _,self.theta_map,self.phi_map,self.hh=vc_estimator_map(phi,theta,nsidegw,nsidedip,hh=None,shuffle=shuffle)
        theta_pix,phi_pix= hp.pix2ang(nsidegw,np.arange(0,len(self.hh),1))
        self.n_gw=np.array([np.sin(theta_pix)*np.cos(phi_pix),np.sin(theta_pix)*np.sin(phi_pix),np.cos(theta_pix)])        
        super().__init__(parameters={ll: None for ll in ['theta_dipole','phi_dipole','Nmono','vocalpha']})
        
    def log_likelihood(self):       
        
        theta_dipole,phi_dipole= self.parameters['theta_dipole'],self.parameters['phi_dipole']
        n_dipole=np.array([np.sin(theta_dipole)*np.cos(phi_dipole),np.sin(theta_dipole)*np.sin(phi_dipole),np.cos(theta_dipole)])
        dipole_projection=np.dot(n_dipole,self.n_gw)
        Nexp=(self.parameters['Nmono']/len(self.hh))*(1+self.parameters['vocalpha']*dipole_projection)
        log_likeli= np.sum(-Nexp+self.hh*np.log(Nexp))
        
        return log_likeli
    
    
def phitheta2skymap(phi,theta,nside):
    '''
    Converts the phi ad theta variables to a skymap
    
    '''
    npixels=hp.nside2npix(nside)
    dOmega_sterad=4*xp.pi/npixels
    dOmega_deg2=xp.power(180/xp.pi,2.)*dOmega_sterad
    indices = hp.ang2pix(nside, theta, phi)
    counts_map = xp.zeros(npixels)
    for indx in range(npixels):
        ind=xp.where(indices==indx)[0]
        counts_map[indx]=len(ind)
        if ind.size==0:
            continue
    counts_map/=(len(theta)*dOmega_sterad)
    return counts_map, dOmega_sterad

def skymapcontours(skymap,CI):
    '''
    Draw the skymap contours
    '''
    skymap[skymap==hp.UNSEEN]=0.
    skymap/=np.sum(skymap)
    sortedmap=np.sort(skymap)[::-1]
    sortedindex=np.argsort(skymap)[::-1]
    cumulativedistro=np.cumsum(sortedmap)
    idx=np.where(cumulativedistro<=CI)[0]
    totake=sortedindex[:idx[-1]]
    
    toret=np.zeros_like(skymap)
    toret[totake]=1
    toret[toret==0]=hp.UNSEEN
    return toret




