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
    A class to handle a list of detected GW signals from simulations. This can be used to
    evaluate selection effects or detection expectations under some priors
    """

    def __init__(self,m1d,m2d,dl,prior_vals,snr_det,snr_cut,ifar,ifar_cut,ntotal,Tobs,condition_check=False):

        """
        This class is used to manage a list of detected injections to calculated
        GW selection effects. This class uses injections which are given in source frame.

        Parameters
        ----------
        file_injections: string (optional)
            File containing the injections ICAROGW format
        m1d: _np.arrray
            Mass 1 detector frame of detected events (provide if file_injection is not provided)
        m2d: _np.arrray
            Mass 2 detector frame of detected events (provide if file_injection is not provided)
        dl: _np.arrray
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
        self.condition_check=condition_check
        self.snr_cut = snr_cut
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

        #Sub-sample the selected injections in order to reduce the computational load
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
        log_jacobian_term = _np.log(_np.abs(_detector_to_source_jacobian(self.z_samples, self.new_cosmo,dl=self.dldet)))
        self.log_weights_astro = log_numer-_np.log(self.ini_prior)-log_jacobian_term
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
        """
        Calculates and returns the effective injections contributing to the calculation of the selection effect.
        See Eq 9 of (https://arxiv.org/pdf/1904.10879.pdf) for more details
        """
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

    def simulate_moc(self, n_moc = 100, Nsamp=5000, mode_likelihood='delta', filepath='.'):
        """
        Will simulate a set of injections using the various likelihood estimators.
        The code automatically put a :math:`d^2` prior insde

        Parameters
        ----------
        n_moc : integer
            The number of moc injeections that you want to simulate
        Nsamp : integer
            The number of samples that you want for every posterior
        mode_likelihood: string
            either 'delta' to simulate delta-like posterior samples or 'uniform'
            to simulate posterior samples distributed according to a uniform prior in masses (detector)
            and d_l**2 prior
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
        self.log_weights_astro = log_numer-_np.log(self.ini_prior)-log_jacobian_term
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
        """
        Calculates and returns the effective injections contributing to the calculation of the selection effect.
        See Eq 9 of (https://arxiv.org/pdf/1904.10879.pdf) for more details
        """
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

    def simulate_moc(self, n_moc = 100, Nsamp=5000, mode_likelihood='delta', filepath='.'):
        """
        Will simulate a set of injections using the various likelihood estimators.
        The code automatically put a :math:`d^2` prior insde

        Parameters
        ----------
        n_moc : integer
            The number of moc injeections that you want to simulate
        Nsamp : integer
            The number of samples that you want for every posterior
        mode_likelihood: string
            either 'delta' to simulate delta-like posterior samples or 'uniform'
            to simulate posterior samples distributed according to a uniform prior in masses (detector)
            and d_l**2 prior
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

            _np.savez(filepath+'injection_{:d}.moc'.format(i),dl=d_uni[sub_index],md1=md1_uni[sub_index],md2=md2_uni[sub_index],
            dl_true=dl[i],md1_true=md1[i],md2_true=md2[i],snr_det=det_snr[i],z_true=zz[i],ms1_true=ms1[i],ms2_true=ms2[i])
