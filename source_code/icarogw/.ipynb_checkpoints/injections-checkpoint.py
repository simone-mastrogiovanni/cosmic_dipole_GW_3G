"""
Module for managing Injections and calculate selection effects.
"""

import numpy as _np
import copy as _copy
import sys as _sys
import h5py as _h5py
import pickle as _pickle
from .utils.conversions import source_frame_to_detector_frame as _source_frame_to_detector_frame
from .utils.conversions import detector_frame_to_source_frame as _detector_frame_to_source_frame
from .utils.conversions import detector_to_source_jacobian as _detector_to_source_jacobian
import bilby as _bilby
from astropy import units as _u
from scipy.special import logsumexp as _logsumexp


__all__=['injections_at_detector','injections_at_source']

class injections_at_detector():
    """
    BLA
    """

    def __init__(self,m1d,m2d,dl,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False):

        """
        This class is used to manage a list of detected injections to calculated
        GW selection effects. This class uses injections which are given in source frame.

        Parameters
        ----------
        file_injections: string (optional)
            File containing the injections ICAROGW format
        cosmo_ref: fast_cosmology class
            Cosmology class corresponding to the cosmology used for injections
        m1s: _np.arrray
            Mass 1 source frame of detected events (provide if file_injection is not provided)
        m2s: _np.arrray
            Mass 2 source frame of detected events (provide if file_injection is not provided)
        z: _np.arrray
            redshift of detected events (provide if file_injection is not provided)
        prior_vals: _np.arrray
            Used prior draws for inejctions (provide if file_injection is not provided)
        snr_det: _np.arrray
            SNR of detected events (provide if file_injection is not provided)
        snr_cut: float
            Set different to 0 if you wanto to apply a different SNR cut.
        ntotal: float
            Total number of simulated injections (detected and not). This is necessary to compute the expected number of detections
        Tobs: float
            Lenght of time for the run in years (used to calculate rates)
        """

        self.snr_cut = snr_cut
        self.condition_check=condition_check
        self.ntotal=ntotal
        self.ntotal_original=ntotal
        self.dl_original=dl
        self.m1d_original=m1d
        self.m2d_original=m2d
        self.snr_original=snr_det
        self.ini_prior_original=prior_vals
        self.ifar=ifar
        self.ifar_cut=ifar_cut
        self.Tobs=Tobs

        idet=_np.where((self.snr_original>snr_cut) & (self.ifar>self.ifar_cut))[0]

        self.idet=idet
        self.m1det=m1d[idet]
        self.m2det=m2d[idet]
        self.dldet=dl[idet]
        self.snrdet=self.snr_original[idet]
        self.ini_prior=self.ini_prior_original[idet]

    def update_cut(self,snr_cut=0,ifar_cut=0,fraction=None):
        print('Selecting injections with SNR {:f} and IFAR {:f} yr'.format(snr_cut,ifar_cut))

        self.snr_cut=snr_cut
        self.ifar_cut=ifar_cut

        idet=_np.where((self.snr_original>snr_cut) & (self.ifar>self.ifar_cut))[0]
        if fraction is not None:
            idet=_np.random.choice(idet,size=int(len(idet)/fraction),replace=False)
            self.ntotal=int(self.ntotal_original/fraction)
            print('Working with a total of {:d} injections'.format(len(idet)))


        self.idet=idet
        self.m1det=self.m1d_original[idet]
        self.m2det=self.m2d_original[idet]
        self.dldet=self.dl_original[idet]
        self.snrdet=self.snr_original[idet]
        self.ini_prior=self.ini_prior_original[idet]

    def return_reweighted_injections(self, m_prior, z_prior, new_samples = 5000):
        """
        This method returns the injections reweighted with a set of new priors.

        Parameters
        ----------
        new_cosmo : cosmology class
            Cosmology class from the cosmologies module
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
            redshift prior module from the prior.redshift module
        new_samples : int, default=5000
            number of posterior samples to return

        Returns
        -------
        ms1 : numpy.array
            Array corresponding to mass_1 in Msol at source frame
        ms2 : numpy.array
            Array corresponding to mass_2 in Msol at source frame
        z_samples : numpy.array
            Array corresponding to redshift
        total_weight : float
            total_weight assigned by the new priors.
        new_index: np.array
            index used for the resampling
        """

        new_cosmo = z_prior.cosmo

        ms1, ms2, z_samples = _detector_frame_to_source_frame(new_cosmo,self.m1det,self.m2det,self.dldet)

        if self.condition_check:
            mass_a = _np.hstack([ms1,ms2])
            if (_np.max(mass_a)<m_prior.mmax) | (_np.min(mass_a)>m_prior.mmin):
                print('The  injections source frame masses are not enough to cover all the prior range')
                print('Masses prior range {:.2f}-{:.2f} Msol'.format(m_prior.mmin,m_prior.mmax))
                print('Injections range {:.2f}-{:.2f} Msol'.format(_np.min(mass_a),_np.max(mass_a)))
                _sys.exit()

        log_jacob_prior = _np.log(_np.abs(_detector_to_source_jacobian(z_samples, new_cosmo)))+_np.log(self.ini_prior)
        log_weights = m_prior.log_joint_prob(ms1,ms2)+z_prior.log_prob(z_samples)-log_jacob_prior
        total_weight=_np.exp(_logsumexp(log_weights))
        weights = _np.exp(log_weights)/total_weight
        #Draw the new PE samples according to the new weights
        new_index = _np.random.choice(len(z_samples),size=new_samples,p=weights,replace=True)

        return ms1[new_index], ms2[new_index], z_samples[new_index] , total_weight, new_index

    def calculate_Neff(self,m_prior,z_prior):

        """
        Calculates the effective number of samples contributing to the Montecarlo draws
        """

        new_cosmo = z_prior.cosmo
        ms1, ms2, z_samples = _detector_frame_to_source_frame(new_cosmo,self.m1det,self.m2det,self.dldet)

        # Checks if the injections covers the entire prior range. If not, throws an errror if the flag is True
        if self.condition_check:
            mass_a = _np.hstack([ms1,ms2])
            if (_np.max(mass_a)<m_prior.mmax) | (_np.min(mass_a)>m_prior.mmin):
                print('The  injections source frame masses are not enough to cover all the prior range')
                print('Masses prior range {:.2f}-{:.2f} Msol'.format(m_prior.mmin,m_prior.mmax))
                print('Injections range {:.2f}-{:.2f} Msol'.format(_np.min(mass_a),_np.max(mass_a)))
                _sys.exit()

        # Calculate the weights according to Eq. 18 in the document
        log_numer = z_prior.log_prob(z_samples)+m_prior.log_joint_prob(ms1,ms2)
        log_jacobian_term = _np.log(_np.abs(_detector_to_source_jacobian(z_samples, new_cosmo)))
        # We remove the original prior and the spin prior
        log_weights = _np.log(4)+log_numer-_np.log(self.ini_prior)-log_jacobian_term
        mean = _np.exp(_logsumexp(log_weights))/self.ntotal
        var = _np.exp(_logsumexp(log_weights*2))/(self.ntotal**2)-(mean**2)/self.ntotal

        return (mean**2)/var

    def expected_number_detection(self,R0, m_prior,z_prior,Tobs=None):
        """
        This method will return the expected number of GW detection given the injection set.

        Parameters
        ----------
        R0 : float
            Merger rate in comoving volume in Gpc-3yr-1
        Tobs : float
            Number of years in which detectors were observing.
        new_cosmo : cosmology class
            Cosmology class from the cosmologies module
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
            redshift prior module from the prior.redshift module
        """
        new_cosmo = z_prior.cosmo

        if Tobs is None:
            Tobs=self.Tobs

        if self.ntotal is None:
            raise ValueError('Please provide the total number of simulated injections')

        ms1, ms2, z_samples = _detector_frame_to_source_frame(new_cosmo,self.m1det,self.m2det,self.dldet)

        if self.condition_check:

            mass_a = _np.hstack([ms1,ms2])
            if (_np.max(mass_a)<m_prior.mmax) | (_np.min(mass_a)>m_prior.mmin):
                print('The  injections source frame masses are not enough to cover all the prior range')
                print('Masses prior range {:.2f}-{:.2f} Msol'.format(m_prior.mmin,m_prior.mmax))
                print('Injections range {:.2f}-{:.2f} Msol'.format(_np.min(mass_a),_np.max(mass_a)))

                _sys.exit()

        log_jacob_prior = _np.log(_np.abs(_detector_to_source_jacobian(z_samples, new_cosmo)))+_np.log(self.ini_prior)
        #0.25 is coming from the spins, 1/2 on one spins, 1/2 on the other one
        log_to_sum = _np.log(0.25)+m_prior.log_joint_prob(ms1,ms2)+z_prior.log_prob_astro(z_samples)-log_jacob_prior

        return R0*Tobs*_np.exp(_logsumexp(log_to_sum))/self.ntotal

    def gw_only_selection_effect(self, m_prior, z_prior):
        """
        Will evaluate the GW-only selection effect using the set of injections

        Parameters
        ----------
        new_cosmo : cosmology clasnp.exp(0.25)s
            Cosmology class from the cosmologies module
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
            redshift prior module from the prior.redshift module

        Returns
        -------
        Selection effect (float)
        """

        new_cosmo = z_prior.cosmo

        ms1, ms2, z_samples = _detector_frame_to_source_frame(new_cosmo,self.m1det,self.m2det,self.dldet)

        if self.condition_check:
            mass_a = _np.hstack([ms1,ms2])
            if (_np.max(mass_a)<m_prior.mmax) | (_np.min(mass_a)>m_prior.mmin):
                print('The  injections source frame masses are not enough to cover all the prior range')
                print('Masses prior range {:.2f}-{:.2f} Msol'.format(m_prior.mmin,m_prior.mmax))
                print('Injections range {:.2f}-{:.2f} Msol'.format(_np.min(mass_a),_np.max(mass_a)))

                _sys.exit()

        log_jacob_prior = _np.log(_np.abs(_detector_to_source_jacobian(z_samples, new_cosmo)))+_np.log(self.ini_prior)
        #0.25 is coming from the spins, 1/2 on one spins, 1/2 on the other one
        log_to_sum = _np.log(0.25)+m_prior.log_joint_prob(ms1,ms2)+z_prior.log_prob(z_samples)-log_jacob_prior
        return _np.exp(_logsumexp(log_to_sum))/self.ntotal

    def simulate_moc(self, m_prior, z_prior, n_moc = 100, Nsamp=5000, mode_likelihood='quick', filepath='.'):
        """
        Will simulate a set of injections using the various likelihood estimators.
        The code automatically put a :math:`d^2` prior insde

        Parameters
        ----------
        new_cosmo : cosmology class
            Cosmology class from the cosmologies module
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
            redshift prior module from the prior.redshift module
        n_moc : int
            How many posterior samples you want
        Nsamp : int
            How many samples per posterior
        filepath : str (optional)
            Where to save the injections

        """

        new_cosmo = _copy.deepcopy(z_prior.cosmo)

        # Reweight the injections to mimic the distribution you want to inject
        ms1,ms2, zz, _, index =  self.return_reweighted_injections(m_prior, z_prior, new_samples = n_moc)
        md1,md2,dl = _source_frame_to_detector_frame(new_cosmo,ms1,ms2,zz)

        eta = md1*md2/_np.power(md1+md2,2)
        chirp_mass = (md1+md2)*_np.power(eta,3./5)

        det_snr = self.snrdet[index]

        dlmax=_np.max(self.dldet)
        md1max=_np.max(self.m1det)
        md2max=_np.max(self.m2det)

        for i in range(n_moc):
            print('Generating injection {:d}'.format(i))
            print('The injection has m1d {:.2f} Msol, m2d {:.2f} Msol, dl {:.0f} Mpc and SNR {:.1f}'.format(md1[i],md2[i],dl[i],det_snr[i]))


            if mode_likelihood=='uniform':
                d_uni = _np.random.uniform(low=0,
                high=dlmax,size=Nsamp*500)
                md1_uni = _np.random.uniform(low=0,high=md1max,size=Nsamp*500)
                md2_uni = _np.random.uniform(low=0,high=md1max,size=Nsamp*500)
                for j in range(Nsamp*500):
                    while (md1_uni[j]) < (md2_uni[j]):
                        md1_uni[j] = _np.random.uniform(low=0,high=md1max)
                        md2_uni[j] = _np.random.uniform(low=0,high=md1max)

                weights = _np.power(d_uni,2)
                weights /= _np.sum(weights)

                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)

            elif mode_likelihood=='delta':

                d_uni=_np.array([dl[i]])
                md1_uni=_np.array([md1[i]])
                md2_uni=_np.array([md2[i]])
                sub_index=_np.array([0,0])

            elif mode_likelihood=='test':

                likeli_eta = _bilby.core.prior.Gaussian(mu=eta[i],sigma=0.022*8/det_snr[i])
                likeli_chirp_mass = _bilby.core.prior.Gaussian(mu=chirp_mass[i],sigma=chirp_mass[i]*0.08*8/det_snr[i])
                likeli_distance = _bilby.core.prior.Gaussian(mu=dl[i],sigma=3.6*dl[i]/det_snr[i])

                #Extract observed values of eta, chirp_mass and distance
                noise_eta = likeli_eta.sample()
                noise_chirp_mass = likeli_chirp_mass.sample()
                noise_distance = likeli_distance.sample()

                # Define the posterior distribution around the observed values
                likeli_eta = _bilby.core.prior.TruncatedGaussian(mu=noise_eta,sigma=0.022*8/det_snr[i],minimum=0,maximum=0.25)
                likeli_chirp_mass = _bilby.core.prior.TruncatedGaussian(mu=noise_chirp_mass,sigma=chirp_mass[i]*0.08*8/det_snr[i],minimum=0,maximum=_np.inf)
                likeli_distance = _bilby.core.prior.TruncatedGaussian(mu=noise_distance,sigma=3.6*dl[i]/det_snr[i],minimum=0,maximum=_np.inf)

                eta_samples = likeli_eta.sample(Nsamp*100)
                chirp_mass_samples = likeli_chirp_mass.sample(Nsamp*100)
                d_uni = likeli_distance.sample(Nsamp*100)

                qvalue= ((1./(2*eta_samples))-1)-(0.5/eta_samples)*_np.sqrt(1-4*eta_samples)
                md1_uni=chirp_mass_samples*_np.power(1+qvalue,1./5)/_np.power(qvalue,3./5)
                md2_uni=qvalue*md1_uni

                # Define weights to generate the posterior samples (using a dl^2 prior)
                jacob=_np.abs(((1-qvalue)/_np.power(1+qvalue,3.))*(chirp_mass_samples/_np.power(md1_uni,2.)))
                weights = _np.power(d_uni,2)/jacob
                weights /=weights.sum()

                # Generate posterio samples below
                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)

            elif mode_likelihood=='logs':

                likeli_eta = _bilby.core.prior.Gaussian(mu=eta[i],sigma=0.022*8/det_snr[i])
                likeli_log_chirp_mass = _bilby.core.prior.Gaussian(mu=_np.log(chirp_mass[i]),sigma=0.08*8/det_snr[i])
                likeli_log_distance = _bilby.core.prior.Gaussian(mu=_np.log(dl[i]),sigma=0.1)

                #Extract observed values of eta, chirp_mass and distance
                noise_eta = likeli_eta.sample()
                noise_log_chirp_mass = likeli_log_chirp_mass.sample()
                noise_log_distance = likeli_log_distance.sample()

                # Define the posterior distribution around the observed values
                likeli_eta = _bilby.core.prior.TruncatedGaussian(mu=noise_eta,sigma=0.022*8/det_snr[i],minimum=0,maximum=0.25)
                likeli_log_chirp_mass = _bilby.core.prior.Gaussian(mu=noise_log_chirp_mass,sigma=0.08*8/det_snr[i])
                likeli_log_distance = _bilby.core.prior.Gaussian(mu=noise_log_distance,sigma=0.1)

                eta_samples = likeli_eta.sample(Nsamp*100)
                log_chirp_mass_samples = likeli_log_chirp_mass.sample(Nsamp*100)
                log_d_uni = likeli_log_distance.sample(Nsamp*100)

                chirp_mass_samples = _np.exp(log_chirp_mass_samples)
                d_uni = _np.exp(log_d_uni)

                qvalue= ((1./(2*eta_samples))-1)-(0.5/eta_samples)*_np.sqrt(1-4*eta_samples)
                md1_uni= chirp_mass_samples*_np.power(1+qvalue,1./5)/_np.power(qvalue,3./5)
                md2_uni=qvalue*md1_uni

                # Define weights to generate the posterior samples (using a dl^2 prior)
                jacob=_np.abs(((1-qvalue)/_np.power(1+qvalue,3.))*(1/_np.power(md1_uni,2.))*(1./d_uni))
                weights = _np.power(d_uni,2)/jacob
                weights /=weights.sum()

                # Generate posterio samples below
                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)


            _np.savez(filepath+'injection_{:d}.moc'.format(i),dl=d_uni[sub_index],md1=md1_uni[sub_index],md2=md2_uni[sub_index],
            dl_true=dl[i],md1_true=md1[i],md2_true=md2[i],snr_det=det_snr[i],z_true=zz[i],ms1_true=ms1[i],ms2_true=ms2[i])


class injections_at_source():
    """
    A class to handle a list of detected GW signals from simulations. This can be used to
    evaluate selection effects or detection expectations under some priors
    """

    def __init__(self,cosmo_ref,m1s,m2s,z,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False):

        """
        This class is used to manage a list of detected injections to calculated
        GW selection effects. This class uses injections which are given in source frame.

        Parameters
        ----------
        file_injections: string (optional)
            File containing the injections ICAROGW format
        cosmo_ref: fast_cosmology class
            Cosmology class corresponding to the cosmology used for injections
        m1s: _np.arrray
            Mass 1 source frame of detected events (provide if file_injection is not provided)
        m2s: _np.arrray
            Mass 2 source frame of detected events (provide if file_injection is not provided)
        z: _np.arrray
            redshift of detected events (provide if file_injection is not provided)
        prior_vals: _np.arrray
            Used prior draws for inejctions (provide if file_injection is not provided)
        snr_det: _np.arrray
            SNR of detected events (provide if file_injection is not provided)
        snr_cut: float
            Set different to 0 if you wanto to apply a different SNR cut.
        ntotal: float
            Total number of simulated injections (detected and not). This is necessary to compute the expected number of detections
        Tobs: float
            Lenght of time for the run in years (used to calculate rates)
        """
        # Saves what you provided in the class
        self.cosmo_ref = _copy.deepcopy(cosmo_ref)
        self.condition_check=condition_check
        self.snr_cut = snr_cut
        self.ntotal=ntotal
        self.ntotal_original=ntotal
        self.z_original=z
        self.m1s_original=m1s
        self.m2s_original=m2s
        self.snr_original=snr_det
        self.ini_prior_original=prior_vals
        self.ifar=ifar
        self.ifar_cut=ifar_cut
        self.Tobs=Tobs
        # Convert from source frame to detector frame and select injections according to SNR and IFAR
        md1, md2, dl = _source_frame_to_detector_frame(self.cosmo_ref,self.m1s_original,self.m2s_original,self.z_original)
        idet=_np.where((self.snr_original>snr_cut) & (self.ifar>self.ifar_cut))[0]

        self.idet=idet
        self.m1det=md1[idet]
        self.m2det=md2[idet]
        self.dldet=dl[idet]
        self.snrdet=self.snr_original[idet]
        self.ini_prior=self.ini_prior_original[idet]

        self.log_origin_jacobian = _np.log(_np.abs(_detector_to_source_jacobian(self.z_original[self.idet],self.cosmo_ref,dl=self.dldet)))

    def update_cut(self,snr_cut=0,ifar_cut=0,fraction=None):
        print('Selecting injections with SNR {:f} and IFAR {:f} yr'.format(snr_cut,ifar_cut))

        self.snr_cut=snr_cut
        self.ifar_cut=ifar_cut

        # Convert from source frame to detector frame and select injections according to SNR and IFAR
        md1, md2, dl = _source_frame_to_detector_frame(self.cosmo_ref,self.m1s_original,self.m2s_original,self.z_original)
        idet=_np.where((self.snr_original>snr_cut) & (self.ifar>self.ifar_cut))[0]

        #Sub-sample the selected injections in order to reduce the computational load
        if fraction is not None:
            idet=_np.random.choice(idet,size=int(len(idet)/fraction),replace=False)
            self.ntotal=int(self.ntotal_original/fraction)
            print('Working with a total of {:d} injections'.format(len(idet)))

        self.idet=idet
        self.m1det=md1[idet]
        self.m2det=md2[idet]
        self.dldet=dl[idet]
        self.snrdet=self.snr_original[idet]
        self.ini_prior=self.ini_prior_original[idet]
        self.log_origin_jacobian = _np.log(_np.abs(_detector_to_source_jacobian(self.z_original[self.idet],self.cosmo_ref,dl=self.dldet)))

    def update_VT(self,m_prior,z_prior):
        """
        This method updates the sensitivity estimations.

        Parameters
        ----------
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
            redshift prior module from the prior.redshift module
        """

        self.new_cosmo = _copy.deepcopy(z_prior.cosmo)
        self.ms1, self.ms2, self.z_samples = _detector_frame_to_source_frame(self.new_cosmo,self.m1det,self.m2det,self.dldet)

        # Checks if the injections covers the entire prior range. If not, throws an errror if the flag is True
        if self.condition_check:
            mass_a = _np.hstack([ms1,ms2])
            if (_np.max(mass_a)<m_prior.mmax) | (_np.min(mass_a)>m_prior.mmin):
                print('The  injections source frame masses are not enough to cover all the prior range')
                print('Masses prior range {:.2f}-{:.2f} Msol'.format(m_prior.mmin,m_prior.mmax))
                print('Injections range {:.2f}-{:.2f} Msol'.format(_np.min(mass_a),_np.max(mass_a)))
                _sys.exit()

        # Calculate the weights according to Eq. 18 in the document
        log_numer = z_prior.log_prob_astro(self.z_samples)+m_prior.log_joint_prob(self.ms1,self.ms2)
        log_jacobian_term = _np.log(_np.abs(_detector_to_source_jacobian(self.z_samples, self.new_cosmo,dl=self.dldet)))-self.log_origin_jacobian
        # We have to put a 0.25 for the spin prior
        self.log_weights_astro = _np.log(0.25)+log_numer-_np.log(self.ini_prior)-log_jacobian_term
        self.log_weights = self.log_weights_astro - _np.log(z_prior.norm_fact)
        # This is the Volume-Time in which we expect to detect. You can multiply it by R_0 Tobs to get the expected number of detections in Gpc^3 yr
        self.VT_sens=_np.exp(_logsumexp(self.log_weights_astro))/self.ntotal
        # This is the fraction of events we expect to detect, a.k.a. the selection effect
        self.VT_fraction=self.VT_sens/z_prior.norm_fact


    def return_reweighted_injections(self,new_samples = 5000):
        """
        This method returns the injections reweighted with a set of new priors.

        Parameters
        ----------
        new_samples : int, default=5000
            number of posterior samples to return

        Returns
        -------
        ms1 : numpy.array
            Array corresponding to mass_1 in Msol at source frame
        ms2 : numpy.array
            Array corresponding to mass_2 in Msol at source frame
        z_samples : numpy.array
            Array corresponding to redshift
        total_weight : float
            total_weight assigned by the new priors.
        new_index: np.array
            index used for the resampling
        """

        total_weight=_np.exp(_logsumexp(self.log_weights))
        weights = _np.exp(self.log_weights)/total_weight
        #Draw the new PE samples according to the new weights
        new_index = _np.random.choice(len(self.z_samples),size=new_samples,p=weights,replace=True)
        return self.ms1[new_index], self.ms2[new_index], self.z_samples[new_index] , total_weight, new_index

    def calculate_Neff(self):
        mean = self.VT_fraction
        var = _np.exp(_logsumexp(self.log_weights*2))/(self.ntotal**2)-(mean**2)/self.ntotal
        return (mean**2)/var

    def expected_number_detection(self,R0):
        """
        This method will return the expected number of GW detection given the injection set. Tobs is automatically saved in the class creation

        Parameters
        ----------
        R0 : float
            Merger rate in comoving volume in Gpc-3yr-1
        """
        return self.VT_sens*R0*self.Tobs

    def gw_only_selection_effect(self):
        """
        Will evaluate the GW-only selection effect using the set of injections

        Returns
        -------
        Selection effect (float)
        """
        return self.VT_fraction

    def simulate_moc(self, n_moc = 100, Nsamp=5000, mode_likelihood='logs', filepath='.'):
        """
        Will simulate a set of injections using the various likelihood estimators.
        The code automatically put a :math:`d^2` prior insde

        Parameters
        ----------
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
            redshift prior module from the prior.redshift module
        n_moc : int
            How many posterior samples you want
        Nsamp : int
            How many samples per posterior
        filepath : str (optional)
            Where to save the injections

        """


        # Reweight the injections to mimic the distribution you want to inject
        ms1,ms2, zz, _, index =  self.return_reweighted_injections(new_samples = n_moc)
        md1,md2,dl = _source_frame_to_detector_frame(self.new_cosmo,ms1,ms2,zz)

        eta = md1*md2/_np.power(md1+md2,2)
        chirp_mass = (md1+md2)*_np.power(eta,3./5)

        det_snr = self.snrdet[index]

        dlmax=_np.max(self.dldet)
        md1max=_np.max(self.m1det)
        md2max=_np.max(self.m2det)

        for i in range(n_moc):
            print('Generating injection {:d}'.format(i))
            print('The injection has m1d {:.2f} Msol, m2d {:.2f} Msol, dl {:.0f} Mpc and SNR {:.1f}'.format(md1[i],md2[i],dl[i],det_snr[i]))


            if mode_likelihood=='uniform':
                d_uni = _np.random.uniform(low=0,
                high=dlmax,size=Nsamp*500)
                md1_uni = _np.random.uniform(low=0,high=md1max,size=Nsamp*500)
                md2_uni = _np.random.uniform(low=0,high=md1max,size=Nsamp*500)
                for j in range(Nsamp*500):
                    while (md1_uni[j]) < (md2_uni[j]):
                        md1_uni[j] = _np.random.uniform(low=0,high=md1max)
                        md2_uni[j] = _np.random.uniform(low=0,high=md1max)

                weights = _np.power(d_uni,2)
                weights /= _np.sum(weights)

                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)

            elif mode_likelihood=='delta':

                d_uni=_np.array([dl[i]])
                md1_uni=_np.array([md1[i]])
                md2_uni=_np.array([md2[i]])
                sub_index=_np.array([0,0])

            elif mode_likelihood=='test':

                likeli_eta = _bilby.core.prior.TruncatedGaussian(mu=eta[i],sigma=0.022*8/det_snr[i],minimum=0,maximum=0.25)
                likeli_chirp_mass = _bilby.core.prior.TruncatedGaussian(mu=chirp_mass[i],sigma=chirp_mass[i]*0.08*8/det_snr[i],minimum=0,maximum=_np.inf)
                likeli_distance = _bilby.core.prior.TruncatedGaussian(mu=dl[i],sigma=3.6*dl[i]/det_snr[i],minimum=0,maximum=_np.inf)

                #Extract observed values of eta, chirp_mass and distance
                noise_eta = likeli_eta.sample()
                noise_chirp_mass = likeli_chirp_mass.sample()
                noise_distance = likeli_distance.sample()

                # Define the posterior distribution around the observed values
                likeli_eta = _bilby.core.prior.TruncatedGaussian(mu=noise_eta,sigma=0.022*8/det_snr[i],minimum=0,maximum=0.25)
                likeli_chirp_mass = _bilby.core.prior.TruncatedGaussian(mu=noise_chirp_mass,sigma=chirp_mass[i]*0.08*8/det_snr[i],minimum=0,maximum=_np.inf)
                likeli_distance = _bilby.core.prior.TruncatedGaussian(mu=noise_distance,sigma=3.6*dl[i]/det_snr[i],minimum=0,maximum=_np.inf)

                eta_samples = likeli_eta.sample(Nsamp*100)
                chirp_mass_samples = likeli_chirp_mass.sample(Nsamp*100)
                d_uni = likeli_distance.sample(Nsamp*100)

                qvalue= ((1./(2*eta_samples))-1)-(0.5/eta_samples)*_np.sqrt(1-4*eta_samples)
                md1_uni=chirp_mass_samples*_np.power(1+qvalue,1./5)/_np.power(qvalue,3./5)
                md2_uni=qvalue*md1_uni

                # Define weights to generate the posterior samples (using a dl^2 prior)
                jacob=_np.abs(((1-qvalue)/_np.power(1+qvalue,3.))*(chirp_mass_samples/_np.power(md1_uni,2.)))
                weights = _np.power(d_uni,2)/jacob
                weights /=weights.sum()

                # Generate posterio samples below
                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)
            
            elif mode_likelihood=='gaussian':

                likeli_md1 = _bilby.core.prior.Gaussian(mu=md1[i],sigma=0.20*md1[i]*8/det_snr[i])
                likeli_md2 = _bilby.core.prior.Gaussian(mu=md2[i],sigma=0.20*md2[i]*8/det_snr[i])
                likeli_distance = _bilby.core.prior.Gaussian(mu=dl[i],sigma=3.6*dl[i]/det_snr[i])

                #Extract observed values of eta, chirp_mass and distance
                noise_md1 = likeli_md1.sample()
                noise_md2 = likeli_md2.sample()
                noise_distance = likeli_distance.sample()

                # Define the posterior distribution around the observed values
                likeli_md1 = _bilby.core.prior.TruncatedGaussian(mu=noise_md1,sigma=0.20*md1[i]*8/det_snr[i],minimum=0,maximum=_np.inf)
                likeli_md2 = _bilby.core.prior.TruncatedGaussian(mu=noise_md2,sigma=0.20*md2[i]*8/det_snr[i],minimum=0,maximum=_np.inf)
                likeli_distance = _bilby.core.prior.TruncatedGaussian(mu=noise_distance,sigma=3.6*dl[i]/det_snr[i],minimum=0,maximum=_np.inf)

                md1_uni = likeli_md1.sample(Nsamp*100)
                md2_uni = likeli_md2.sample(Nsamp*100)
                d_uni = likeli_distance.sample(Nsamp*100)
                
                pos = _np.where(md1_uni<md2_uni)[0]
                md1_uni[pos], md2_uni[pos] = md2_uni[pos], md1_uni[pos]

                # Define weights to generate the posterior samples (using a dl^2 prior)
                weights = _np.power(d_uni,2)
                weights /=weights.sum()

                # Generate posterio samples below
                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)
                
            elif mode_likelihood=='gaussian_5_percent':

                likeli_md1 = _bilby.core.prior.Gaussian(mu=md1[i],sigma=0.05*md1[i]*12/det_snr[i])
                likeli_md2 = _bilby.core.prior.Gaussian(mu=md2[i],sigma=0.05*md2[i]*12/det_snr[i])
                likeli_distance = _bilby.core.prior.Gaussian(mu=dl[i],sigma=0.05*dl[i]*12/det_snr[i])

                #Extract observed values of eta, chirp_mass and distance
                noise_md1 = likeli_md1.sample()
                noise_md2 = likeli_md2.sample()
                noise_distance = likeli_distance.sample()

                # Define the posterior distribution around the observed values
                likeli_md1 = _bilby.core.prior.TruncatedGaussian(mu=noise_md1,sigma=0.05*md1[i]*12/det_snr[i],minimum=0,maximum=_np.inf)
                likeli_md2 = _bilby.core.prior.TruncatedGaussian(mu=noise_md2,sigma=0.05*md2[i]*12/det_snr[i],minimum=0,maximum=_np.inf)
                likeli_distance = _bilby.core.prior.TruncatedGaussian(mu=noise_distance,sigma=0.05*dl[i]*12/det_snr[i],minimum=0,maximum=_np.inf)

                md1_uni = likeli_md1.sample(Nsamp*100)
                md2_uni = likeli_md2.sample(Nsamp*100)
                d_uni = likeli_distance.sample(Nsamp*100)
                
                pos = _np.where(md1_uni<md2_uni)[0]
                md1_uni[pos], md2_uni[pos] = md2_uni[pos], md1_uni[pos]

                # Define weights to generate the posterior samples (using a dl^2 prior)
                weights = _np.power(d_uni,2)
                weights /=weights.sum()

                # Generate posterio samples below
                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)

            elif mode_likelihood=='logs':

                likeli_eta = _bilby.core.prior.Gaussian(mu=eta[i],sigma=0.022*8/det_snr[i])
                likeli_log_chirp_mass = _bilby.core.prior.Gaussian(mu=_np.log(chirp_mass[i]),sigma=0.08*8/det_snr[i])
                likeli_log_distance = _bilby.core.prior.Gaussian(mu=_np.log(dl[i]),sigma=0.1)

                #Extract observed values of eta, chirp_mass and distance
                noise_eta = likeli_eta.sample()
                noise_log_chirp_mass = likeli_log_chirp_mass.sample()
                noise_log_distance = likeli_log_distance.sample()

                # Define the posterior distribution around the observed values
                likeli_eta = _bilby.core.prior.TruncatedGaussian(mu=noise_eta,sigma=0.022*8/det_snr[i],minimum=0,maximum=0.25)
                likeli_log_chirp_mass = _bilby.core.prior.Gaussian(mu=noise_log_chirp_mass,sigma=0.08*8/det_snr[i])
                likeli_log_distance = _bilby.core.prior.Gaussian(mu=noise_log_distance,sigma=0.1)

                eta_samples = likeli_eta.sample(Nsamp*100)
                log_chirp_mass_samples = likeli_log_chirp_mass.sample(Nsamp*100)
                log_d_uni = likeli_log_distance.sample(Nsamp*100)

                chirp_mass_samples = _np.exp(log_chirp_mass_samples)
                d_uni = _np.exp(log_d_uni)

                qvalue= ((1./(2*eta_samples))-1)-(0.5/eta_samples)*_np.sqrt(1-4*eta_samples)
                md1_uni= chirp_mass_samples*_np.power(1+qvalue,1./5)/_np.power(qvalue,3./5)
                md2_uni=qvalue*md1_uni

                # Define weights to generate the posterior samples (using a dl^2 prior)
                jacob=_np.abs(((1-qvalue)/_np.power(1+qvalue,3.))*(1/_np.power(md1_uni,2.))*(1./d_uni))
                weights = _np.power(d_uni,2)/jacob
                weights /=weights.sum()

                # Generate posterio samples below
                sub_index = _np.random.choice(len(d_uni),size=Nsamp,p=weights,replace=True)


            _np.savez(filepath+'injection_{:d}.moc'.format(i),dl=d_uni[sub_index],md1=md1_uni[sub_index],md2=md2_uni[sub_index],
            dl_true=dl[i],md1_true=md1[i],md2_true=md2[i],snr_det=det_snr[i],z_true=zz[i],ms1_true=ms1[i],ms2_true=ms2[i])
