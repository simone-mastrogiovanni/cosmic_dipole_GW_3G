"""
Module for managing posterior samples.
"""

import numpy as _np
import h5py as _h5py
import copy as _copy
from scipy.stats import gaussian_kde as _gaussian_kde
from .utils.conversions import detector_frame_to_source_frame as _detector_frame_to_source_frame
from .utils.conversions import detector_to_source_jacobian as _detector_to_source_jacobian
from scipy.special import logsumexp as _logsumexp

__all__ = ['posterior_samples']

class posterior_samples(object):
    """
    Class for storing and handling posterior samples from GWs. Implemented hdf5 (O2)
    h5 (O3), moc files from icarogw and custom format. Note that we assume uniform prior on masses at detector frame
    and d^2 prior in the original samples.

    Attributes
    ----------
    mass_1_det : numpy.array
        posterior samples of mass 1 detected at the source frame
    mass_2_det : str
        posterior samples of mass 2 detected at the source frame
    distance : numpy.array
        age of the person
    """

    def __init__(self, filename = None,waveform=None, mass_1_det= None, mass_2_det = None, distance = None):
        """
        Read and initialize the posteiror samples

        Parameters
        ----------
        filename : str, optional
            path of the h5, hdf5, moc file to load
        waveform : str, optional
            What field of or waveform of the posterio samples to read from the h5 or hdf5
        mass_1_det : numpy.array, optional (default=None)
            Array of the posterior samples in detector frame corresponding to mass_1 in Msol, provide it if you dont provide the file
        mass_2_det : numpy.array, optional (default=None)
            Array of the posterior samples in detector frame corresponding to mass_2 in Msol, provide it if you dont provide the file
        distance : numpy.array
            Array of the posterior samples in detector frame corresponding to luminosity distance in Mpc, provide it if you dont provide the file
        """

        self.filename = filename
        self.waveform = waveform

        # Assign the posterior samples in the class
        if filename is None:
            self.mass_1_det = mass_1_det
            self.mass_2_det = mass_2_det
            self.distance = distance
            self.nsamples=len(distance)
        else:
            self._load_posterior_samples()

    def gather_more_samples(self,increase_factor):
        """
        Gather more posterior samples by using a gaussian kernel fit. Use with attention as some
        posteriors are not well fitted by a gaussian kernel

        Parameters
        ----------
        increase_factor: int
            How many more samples you want (e.g. 2,4,8 times)
        """

        if not hasattr(self,'nsamples_original'):
            self.nsamples_original=_copy.deepcopy(self.nsamples)
            self.mass_1_det_original=_copy.deepcopy(self.mass_1_det)
            self.mass_2_det_original=_copy.deepcopy(self.mass_2_det)
            self.distance_original=_copy.deepcopy(self.distance)

        self.nsamples *= increase_factor

        # Fit the posterior samples with a 3D gaussian KDE
        if not hasattr(self,'gkde'):
            self.gkde = _gaussian_kde(_np.vstack([self.mass_1_det,self.mass_2_det,self.distance]))

        # Extract new posterior samples from the gaussian KDE
        new_samps = self.gkde.resample(self.nsamples)
        self.mass_1_det = new_samps[0,:]
        self.mass_2_det = new_samps[1,:]
        self.distance = new_samps[2,:]

    def _load_posterior_samples(self):
        """
        Load the posterior samples file
        """
        # The following functions simply load posterior samples from several file formats
        if self.filename.endswith('.moc.npz'):
            samples = _np.load(self.filename)
            self.mass_1_det = samples['md1']
            self.mass_2_det = samples['md2']
            self.distance = samples['dl']
            self.nsamples = len(self.distance)

        elif self.filename.endswith('.hdf5'):

            if self.waveform is None:
                if self.filename.endswith('GW170817_GWTC-1.hdf5'):
                    waveform = 'IMRPhenomPv2NRT_lowSpin_posterior'
                else:
                    waveform = 'Overall_posterior'

            file = _h5py.File(self.filename, 'r')
            data = file[waveform]
            self.distance = data['luminosity_distance_Mpc']
            self.mass_1_det = data['m1_detector_frame_Msun']
            self.mass_2_det = data['m2_detector_frame_Msun']
            self.nsamples = len(self.distance)
            file.close()

            print("Using "+waveform+" posterior with a total of "+str(len(self.distance))+' samples')

        elif self.filename.endswith('h5'):
            file = _h5py.File(self.filename, 'r')

            if self.waveform is None:
                try:
                    approximant = 'PublicationSamples'
                    data = file[approximant]
                except:
                    # Added to manage the events from GWTC2.1
                    approximant = 'IMRPhenomXPHM'
                    data = file[approximant]

                print("Using "+approximant+" posterior with a total of "+str(len(data['posterior_samples']['luminosity_distance']))+' samples')
            else:
                data = file[self.waveform]
                print("Using "+self.waveform+" posterior with a total of "+str(len(data['posterior_samples']['luminosity_distance']))+' samples')

            self.distance = data['posterior_samples']['luminosity_distance']
            self.mass_1_det = data['posterior_samples']['mass_1']
            self.mass_2_det = data['posterior_samples']['mass_2']
            self.nsamples = len(self.distance)
            file.close()

        else:
            raise NotImplementedError

    def compute_source_frame_samples(self, cosmo):
        """
        This method caculates the posterior samples in source frame

        Parameters
        ----------
        cosmo : cosmology class
            Cosmology class from the cosmologies module

        Returns
        -------
        ms1 : numpy.array
            Array corresponding to mass_1 in Msol at source frame
        ms2 : numpy.array
            Array corresponding to mass_2 in Msol at source frame
        z_samples : numpy.array
            Array corresponding to redshift
        """
        # Convert from det frame to source frame
        ms1, ms2, z_samples = _detector_frame_to_source_frame(cosmo,self.mass_1_det,
        self.mass_2_det,self.distance)
        return ms1, ms2, z_samples

    def _log_jacobian_times_prior(self,zvals,cosmo):
        """
        This method returns the Jacobian from the detector frame to the source frame times the d^2 prior

        Parameters
        ----------
        cosmo : cosmology class
            Cosmology class from the cosmologies module

        Returns
        -------
        ms1 : numpy.array
            Array corresponding to mass_1 in Msol at source frame
        ms2 : numpy.array
            Array corresponding to mass_2 in Msol at source frame
        z_samples : numpy.array
            Array corresponding to redshift
        """
        jacobian = _np.abs(_detector_to_source_jacobian(zvals,cosmo))
        dl = cosmo.dl_at_z(zvals)
        # Divide out 10 Gpc for numerical reasons. Even if priors have different ranges originally,
        # these will enter as overall normalizations.
        return _np.log(jacobian)+2*_np.log(dl)

    def return_reweighted_samples(self, m_prior, z_prior,samples=5000):
        """
        This method returns the posterior samples reweighted with a set of new prior.

        Parameters
        ----------
        m_prior : mass prior class
            mass prior class from the prior.mass module
        z_prior : redshift prior class
            redshift prior module from the prior.redshift module
        samples : int, default=5000
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
        """

        cosmo = z_prior.cosmo
        # Convert from detector frame to source frame
        ms1, ms2, z_samples = _detector_frame_to_source_frame(cosmo,self.mass_1_det,self.mass_2_det,self.distance)
        #Calculate weights according to Eq. 13 on the tex document
        log_weights = m_prior.log_joint_prob(ms1,ms2)+z_prior.log_prob(z_samples)-self._log_jacobian_times_prior(z_samples,cosmo)
        total_weight=_np.exp(_logsumexp(log_weights))
        weights = _np.exp(log_weights)/total_weight
        #Draw the new PE samples according to the new weights
        new_index = _np.random.choice(len(z_samples),size=samples,p=weights,replace=True)
        return ms1[new_index], ms2[new_index], z_samples[new_index] , weights ,total_weight
