import numpy as _np
from astropy import constants as _c
from astropy import units as _u
import pickle as _pickle
from scipy.stats import truncnorm as _truncnorm
from scipy.special import logit, expit
from tqdm import tqdm as _tqdm

from .conversions import source_frame_to_detector_frame as _source_frame_to_detector_frame
from ..injections import injections_at_source as _injections_at_source
import pickle as _pickle

def calculate_injections_mode1(mp,zp,Nsamp,dl8=1000,chirp8=10,SNR_thr=8,outfile=None):
    """
    This function simulates a list of detected binaries using a given source-frame model.

    Parameters
    ----------
    mp: mass class
        Icarogw mass prior class
    zp: redshift class
        Icarogw redshift prior class
    Nsamp: integer
        Number of binaries to simulate
    dl8: float
        Luminosity distance at which you expect an average SNR of 8 in Mpc for m8
    chirp8: float
        Detector-frame mass in which you expect an average SNR of 8 for dl8.
    SNR_thr: float
        SNR threshold

    returns
    -------
    true_quantities: dictionary
        Dictionary with several variables of the detected quantities
    noise_quantities: dictionary
        With several variables of noise quantities
    """

    ms1,ms2 = mp.sample(Nsamp)
    z = zp.sample(Nsamp)
    md1, md2, dl = _source_frame_to_detector_frame(zp.cosmo,ms1,ms2,z)
    print('Generated binaries')

    cmd = _np.power(md1*md2,3./5)*_np.power(md1+md2,-1./5)
    eta =  md1*md2/_np.power(md1+md2,2)

    print('Calculating SNR')
    rho_opt_oriented = 8*_np.power(cmd/chirp8,5./6)*(dl8/dl)
    log_theta = _np.random.randn(Nsamp)*0.3/(1+(rho_opt_oriented/8.))
    rho_opt = rho_opt_oriented*_np.exp(log_theta)

    print('Extracting SNR observed')
    det_snr = _np.random.randn(Nsamp) + rho_opt

    print('Selecting events')
    indx=det_snr>SNR_thr
    md1=md1[indx]
    md2=md2[indx]
    dl=dl[indx]
    rho_opt=rho_opt[indx]
    cmd=cmd[indx]
    eta=eta[indx]
    ms1=ms1[indx]
    ms2=ms2[indx]
    z=z[indx]
    det_snr=det_snr[indx]
    log_theta=log_theta[indx]

    print('Extracting noise quantities')
    noise_eta = _np.random.randn(len(md1))*(8*(0.03)/det_snr)+eta
    noise_log_chirp_mass = _np.random.randn(len(md1))*8*(0.04)/det_snr+_np.log(cmd)
    noise_chirp_mass = _np.exp(noise_log_chirp_mass)


    true_quantities = {'md1':md1,
                    'md2':md2,
                      'dl':dl,
                      'snr':rho_opt,
                      'chirp_mass':cmd,
                      'eta':eta,
                      'ms1':ms1,
                      'ms2':ms2,
                      'z':z,
                      'log_theta':log_theta,
                      'dl8':dl8,
                      'chirp8':chirp8}

    noise_quantities = {'snr':det_snr,
                      'chirp_mass':noise_chirp_mass,
                      'eta':noise_eta,
                       'log_theta':log_theta,
                       'dl8':dl8,
                       'chirp8':chirp8}
    
    if outfile is not None:
        injections = _injections_at_source(cosmo_ref=zp.cosmo,m1s=true_quantities['ms1'],m2s=true_quantities['ms2']
                                                     ,z=true_quantities['z']
    ,prior_vals=mp.joint_prob(true_quantities['ms1'],true_quantities['ms2'])*zp.prob(true_quantities['z']), snr_det=noise_quantities['snr'],
    snr_cut=SNR_thr,ifar_cut=0,ifar=_np.ones_like(noise_quantities['snr'])*_np.inf,
    ntotal=Nsamp,Tobs=1)

        _pickle.dump(injections,  open(outfile+'.inj', "wb" ) )

    return true_quantities, noise_quantities

def generate_PE_mode1(true_quantities,noise_quantities,Ninj,Nsamp,filepath='./'):
    """
    Given a list of noise quantities, generates PE

    Parameters
    ----------
    noise_quantities: dict
        Noise quantitites from the Generate injection mode 1
    Ninj: int
        Number of signals you want to generate
    NSamp: int
        How many posterior samples per signal
    filepath: string
        where to save the injections files.
    """

    for i in _tqdm(range(Ninj)):

        rho_samples = _np.random.randn(Nsamp*100)+ noise_quantities['snr'][i]
        rho_samples[rho_samples<=0.]=0.

        a_eta= (0.-noise_quantities['eta'][i])/(8*(0.03)/noise_quantities['snr'][i])
        b_eta= (0.25-noise_quantities['eta'][i])/(8*(0.03)/noise_quantities['snr'][i])

        samples_eta = _truncnorm.rvs(a_eta,b_eta,size=Nsamp*100)*(8*(0.03)/noise_quantities['snr'][i])+noise_quantities['eta'][i]
        samples_chirp_mass = _np.exp(_np.random.randn(Nsamp*100)*8*(0.04)/noise_quantities['snr'][i]+_np.log(noise_quantities['chirp_mass'][i]))

        samples_dl = (8/(rho_samples/_np.exp(noise_quantities['log_theta'][i])))*_np.power(samples_chirp_mass/noise_quantities['chirp8'],5./6)*noise_quantities['dl8']

        q_samples= ((1./(2*samples_eta))-1)-(0.5/samples_eta)*_np.sqrt(1-4*samples_eta)
        md1_samples=samples_chirp_mass*_np.power(1+q_samples,1./5)/_np.power(q_samples,3./5)
        md2_samples=q_samples*md1_samples

        # The implied prior from masses (lasttwo terms) and from the conversion from rho to dl given rho_det and log_theta
        implied_prior = _np.power(samples_dl,-2.)*_np.power(samples_chirp_mass,5./6)*_np.power(md1_samples,-2.)*_np.abs(((1-q_samples)/_np.power(1+q_samples,3.)))
        weights = _np.power(samples_dl,2.)/implied_prior
        weights/=weights.sum()

        sub_index = _np.random.choice(len(samples_dl),size=Nsamp,p=weights,replace=True)

        _np.savez(filepath+'injection_{:d}.moc'.format(i),dl=samples_dl[sub_index],md1=md1_samples[sub_index],md2=md2_samples[sub_index],
                dl_true=true_quantities['dl'][i],md1_true=true_quantities['md1'][i],md2_true=true_quantities['md2'][i],
                 snr_det=noise_quantities['snr'][i],snr_true=true_quantities['snr'][i],z_true=true_quantities['z'][i],ms1_true=true_quantities['ms1'][i],ms2_true=true_quantities['ms2'][i])
        
        
        
        
        
def calculate_injections_mode2(mp,zp,Nsamp,dl8=1000,chirp8=10,SNR_thr=8,outfile=None):
    """
    This function simulates a list of detected binaries using a given source-frame model.

    Parameters
    ----------
    mp: mass class
        Icarogw mass prior class
    zp: redshift class
        Icarogw redshift prior class
    Nsamp: integer
        Number of binaries to simulate
    dl8: float
        Luminosity distance at which you expect an average SNR of 8 in Mpc for m8
    chirp8: float
        Detector-frame mass in which you expect an average SNR of 8 for dl8.
    SNR_thr: float
        SNR threshold

    returns
    -------
    true_quantities: dictionary
        Dictionary with several variables of the detected quantities
    noise_quantities: dictionary
        With several variables of noise quantities
    """
    
    
    ms1,ms2 = mp.sample(Nsamp)
    z = zp.sample(Nsamp)
    md1, md2, dl = _source_frame_to_detector_frame(zp.cosmo,ms1,ms2,z)
    print('Generated binaries')

    cmd = _np.power(md1*md2,3./5)*_np.power(md1+md2,-1./5)
    q =  md2/md1
        
    theta = _np.random.uniform(low=0.,high=1,size=Nsamp)
    
    print('Calculating SNR')
    rho_opt_oriented = 8*_np.power(cmd/chirp8,5./6)*(dl8/dl)
    rho_opt = rho_opt_oriented*theta

    print('Extracting SNR observed')
    det_snr = _np.random.randn(Nsamp) + rho_opt

    print('Selecting events')
    indx=det_snr>SNR_thr
    md1=md1[indx]
    md2=md2[indx]
    dl=dl[indx]
    rho_opt=rho_opt[indx]
    cmd=cmd[indx]
    q=q[indx]
    ms1=ms1[indx]
    ms2=ms2[indx]
    z=z[indx]
    det_snr=det_snr[indx]
    theta=theta[indx]
    
    cmdmin=min(cmd)/1.5
    cmdmax=max(cmd)*1.5
    qmin=min(q)/2
    print(qmin,cmdmin,cmdmax)

    print('Extracting noise quantities')
    
    #a_q= (qmin-q)/(10*0.25*q/det_snr)
    #b_q= (1.-q)/(10*0.25*q/det_snr)
    #noise_q = _truncnorm.rvs(a_q,b_q,size=len(md1))*(10*0.25*q/det_snr)+q
    noise_q = _np.random.randn(len(md1))*(10*0.25*q/det_snr)+q
        
    #a_c= (cmdmin-cmd)/(10*cmd*8e-3/det_snr)
    #b_c= (cmdmax-cmd)/(10*cmd*8e-3/det_snr)
    #noise_chirp_mass = _truncnorm.rvs(a_c,b_c,size=len(md1))*(10*cmd*8e-3/det_snr)+cmd
    noise_chirp_mass = _np.random.randn(len(md1))*10*cmd*8e-3/det_snr+cmd
    
    #a_theta= (0.-theta)/(10*0.3/det_snr)
    #b_theta= (1.-theta)/(10*0.3/det_snr)
    #noise_theta = _truncnorm.rvs(a_theta,b_theta,size=len(md1))*(10*0.3/det_snr)+theta    
    noise_theta = _np.random.randn(len(md1))*10*(0.3)/det_snr+theta

    true_quantities = {'md1':md1,
                    'md2':md2,
                      'dl':dl,
                      'snr':rho_opt,
                      'chirp_mass':cmd,
                      'q':q,
                      'ms1':ms1,
                      'ms2':ms2,
                      'z':z,
                      'theta':theta,
                      'dl8':dl8,
                      'chirp8':chirp8,
                      'cmdmin':cmdmin, 'cmdmax':cmdmax,'qmin':qmin
                      }

    noise_quantities = {'snr':det_snr,
                      'chirp_mass':noise_chirp_mass,
                      'q':noise_q,
                       'theta':noise_theta,
                       'dl8':dl8,
                       'chirp8':chirp8}
    
    if outfile is not None:
        injections = _injections_at_source(cosmo_ref=zp.cosmo,m1s=true_quantities['ms1'],m2s=true_quantities['ms2']
                                                     ,z=true_quantities['z']
    ,prior_vals=mp.joint_prob(true_quantities['ms1'],true_quantities['ms2'])*zp.prob(true_quantities['z']), snr_det=noise_quantities['snr'],
    snr_cut=SNR_thr,ifar_cut=0,ifar=_np.ones_like(noise_quantities['snr'])*_np.inf,
    ntotal=Nsamp,Tobs=1)

        _pickle.dump(injections,  open(outfile+'.inj', "wb" ) )

    return true_quantities, noise_quantities

def generate_PE_mode2(true_quantities,noise_quantities,Ninj,Nsamp,filepath='./'):
    """
    Given a list of noise quantities, generates PE

    Parameters
    ----------
    noise_quantities: dict
        Noise quantitites from the Generate injection mode 1
    Ninj: int
        Number of signals you want to generate
    NSamp: int
        How many posterior samples per signal
    filepath: string
        where to save the injections files.
    """

    for i in _tqdm(range(Ninj)):

        rho_samples = _np.random.randn(Nsamp*100)+ noise_quantities['snr'][i]
        rho_samples[rho_samples<=0.]=0.
        
        a_q= (true_quantities['qmin']-noise_quantities['q'][i])/(10*0.25*true_quantities['q'][i]/noise_quantities['snr'][i])
        b_q= (1.-noise_quantities['q'][i])/(10*0.25*true_quantities['q'][i]/noise_quantities['snr'][i])
        samples_q = _truncnorm.rvs(a_q,b_q,size=Nsamp*100)*(10*0.25*true_quantities['q'][i]/noise_quantities['snr'][i])+noise_quantities['q'][i]
        
        a_c= (true_quantities['cmdmin']-noise_quantities['chirp_mass'][i])/(10*true_quantities['chirp_mass'][i]*8e-3/noise_quantities['snr'][i])
        b_c= (true_quantities['cmdmax']-noise_quantities['chirp_mass'][i])/(10*true_quantities['chirp_mass'][i]*8e-3/noise_quantities['snr'][i])
        samples_chirp_mass = _truncnorm.rvs(a_c,b_c,size=Nsamp*100)*(10*true_quantities['chirp_mass'][i]*8e-3/noise_quantities['snr'][i])+noise_quantities['chirp_mass'][i] 
        
        a_theta= (0.-noise_quantities['theta'][i])/(10*0.3/noise_quantities['snr'][i])
        b_theta= (1.-noise_quantities['theta'][i])/(10*0.3/noise_quantities['snr'][i])
        samples_theta = _truncnorm.rvs(a_theta,b_theta,size=Nsamp*100)*(10*0.3/noise_quantities['snr'][i])+noise_quantities['theta'][i]
        
        samples_dl = samples_theta*(8./rho_samples)*_np.power(samples_chirp_mass/noise_quantities['chirp8'],5./6)*noise_quantities['dl8']

        md1_samples=samples_chirp_mass*_np.power(1+samples_q,1./5)/_np.power(samples_q,3./5)
        md2_samples=samples_q*md1_samples

        # The implied prior from masses (lasttwo terms) and from the conversion from rho to dl given rho_det and log_theta
        implied_prior = _np.power(
            samples_dl,-2.)*_np.power(samples_chirp_mass,5./6)*samples_theta*_np.power(md1_samples,-2.)*samples_chirp_mass
        weights = _np.power(samples_dl,2.)/implied_prior
        weights/=weights.sum()

        sub_index = _np.random.choice(len(samples_dl),size=Nsamp,p=weights,replace=True)

        _np.savez(filepath+'injection_{:d}.moc'.format(i),dl=samples_dl[sub_index],md1=md1_samples[sub_index],md2=md2_samples[sub_index],
                dl_true=true_quantities['dl'][i],md1_true=true_quantities['md1'][i],md2_true=true_quantities['md2'][i],
                 snr_det=noise_quantities['snr'][i],snr_true=true_quantities['snr'][i],z_true=true_quantities['z'][i],ms1_true=true_quantities['ms1'][i],ms2_true=true_quantities['ms2'][i])
