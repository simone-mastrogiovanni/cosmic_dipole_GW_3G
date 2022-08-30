import numpy as _np
import sys as _sys
import bilby as _bilby
import matplotlib.pyplot as _plt
import copy as _copy
from scipy.special import logsumexp as _logsumexp

from scipy.interpolate import splrep as _splrep
from scipy.interpolate import splev as _splev
from scipy.stats import gaussian_kde as _gaussian_kde

from ..cosmologies import flatLCDM as _flatLCDM
from ..cosmologies import w0flatLCDM as _w0flatLCDM
from ..cosmologies import w0waflatLCDM as _w0waflatLCDM

from ..priors.mass import mass_prior as _mass_prior
from ..priors.redshift import redshift_prior as _redshift_prior
from ..utils.conversions import detector_frame_to_source_frame as _detector_frame_to_source_frame
from ..utils.conversions import  detector_to_source_jacobian as _detector_to_source_jacobian
import progressbar



__all__ = ['BBH_likelihood','hierarchical_analysis']

class BBH_likelihood(_bilby.Likelihood):
    """
    Hierarchical likelihood class for Bilby. This class is used to calculate the log likelihood when running Bilby

    Parameters
    ----------
    posterior_samples_dict: dict
        Dictionary of posterior_samples objects from icarogw. The events you want to use
    injections: injection object
        The injection object from icarogw. USed to evaluate selection effects and expected events.
    mass_model: str
        The mass model used, either 'BBH-powerlaw', 'BBH-powerlaw-gaussian','BBH-broken-powerlaw'  and 'BBH-powerlaw-double-gaussian'
    cosmo_model: str
        The cosmological model used, either 'flatLCDM', 'w0flatLCDM' or 'w0waflatLCDM'
    rate_model: str
        The rate evolution model, either 'powerlaw' or 'madau'.
    parallel: integer or None
        If none, the likelihood is evaluated event by event using all posterior samples (slow method).
        If integer built a matrix of posterior samples = parallel for a quick computation.
    ln_evidence: dic
        Log evidence associate to each event. Important if you want to use different waveform models.
        If none, assumed as a constant
    scale_free: bool
        If True it will assume a log uniform prior on the number of events (no poissonian term). Otherwise will use the poissonian terms.
    """

    def __init__(self, posterior_samples_dict, injections,
     mass_model, cosmo_model, rate_model,
     parallel = None, ln_evidences=None,scale_free=True):

        if mass_model not in ['BBH-powerlaw','BBH-powerlaw-gaussian','BBH-broken-powerlaw','BBH-powerlaw-double-gaussian']:
            print('mass model not known')
            _sys.exit()

        # Some initialization below
        self.rate_model=rate_model
        self.posterior_samples_dict = _copy.deepcopy(posterior_samples_dict)
        self.injections = _copy.deepcopy(injections)
        self.parallel=parallel
        self.mass_model=mass_model
        self.cosmo_model=cosmo_model
        self.scale_free=scale_free

        if ln_evidences is None:
            ln_evidences = _np.zeros(len(posterior_samples_dict))

        self.ln_evidences = ln_evidences

        # In this loop we build the matrix of posteriro samples if we  are runnig with the parallel mode
        if parallel is not None:
            len_samples = [len(self.posterior_samples_dict[event].distance)
             for event in list(self.posterior_samples_dict.keys())]
            n_min = _np.min(len_samples)
            n_min = _np.min([n_min,parallel])
            n_ev = len(posterior_samples_dict)

            self.dl_parallel = _np.zeros([n_ev,n_min])
            self.m1det_parallel = _np.zeros([n_ev,n_min])
            self.m2det_parallel = _np.zeros([n_ev,n_min])
            print('Using the parallel mode with {:d} samples'.format(n_min))


            for i,event in enumerate(list(self.posterior_samples_dict.keys())):
                len_single = len(self.posterior_samples_dict[event].distance)
                rand_perm = _np.random.permutation(len_single)

                self.dl_parallel[i,:]=self.posterior_samples_dict[event].distance[rand_perm[:n_min]]
                self.m1det_parallel[i,:]=self.posterior_samples_dict[event].mass_1_det[rand_perm[:n_min]]
                self.m2det_parallel[i,:]=self.posterior_samples_dict[event].mass_2_det[rand_perm[:n_min]]


        # The lines below simply save a list of the parameters for the different population and cosmology models.
        if mass_model == 'BBH-powerlaw':
            self.list_pop_param = ['alpha', 'mmin', 'mmax','beta']
        elif mass_model == 'BBH-powerlaw-gaussian':
            self.list_pop_param = ['alpha', 'mmin', 'mmax','beta','mu_g','sigma_g','lambda_peak','delta_m']
        elif mass_model == 'BBH-broken-powerlaw':
            self.list_pop_param = ['alpha_1','alpha_2', 'mmin', 'mmax','beta','b','delta_m']
        elif mass_model == 'BBH-powerlaw-double-gaussian':
            self.list_pop_param = ['alpha', 'mmin', 'mmax','beta','mu_g_low','sigma_g_low','mu_g_high','sigma_g_high','lambda_g','lambda_g_low','delta_m']

        if cosmo_model == 'flatLCDM':
            self.list_cosmo_param = ['H0','Om0']
        elif cosmo_model == 'w0flatLCDM':
            self.list_cosmo_param = ['H0','Om0','w0']
        elif cosmo_model == 'w0waflatLCDM':
            self.list_cosmo_param = ['H0','Om0','w0','wa']
        else:
            print('Not yet implemented')
            _sys.exit()

        if self.rate_model=='powerlaw':
            self.list_rate_param = ['gamma']
        elif self.rate_model=='madau':
            self.list_rate_param = ['gamma','kappa','zp']

        # Save the total list of parameters
        tot_list = self.list_pop_param+self.list_cosmo_param+self.list_rate_param

        if not self.scale_free:
            tot_list = tot_list+['R0']

        # Initialize the Bilby parameters
        super().__init__(parameters={ll: None for ll in tot_list})

    def log_likelihood(self):
        '''
        Evaluates and return the log-likelihood
        '''

        # initialization of the mass prior for a given set of parameters
        dic = {ll:self.parameters[ll] for ll in self.list_pop_param}
        mp_model = _mass_prior(name=self.mass_model,hyper_params_dict = dic)

        # Initialize the cosmology for a set of parametrs
        if self.cosmo_model=='flatLCDM':
            cosmo = _flatLCDM(Omega_m=self.parameters['Om0'],H0=self.parameters['H0'],astropy_conv=False)
        elif self.cosmo_model=='w0flatLCDM':
            cosmo = _w0flatLCDM(Omega_m=self.parameters['Om0'],H0=self.parameters['H0'],
            w0=self.parameters['w0'],astropy_conv=False)
        elif self.cosmo_model=='w0waflatLCDM':
            cosmo = _w0waflatLCDM(Omega_m=self.parameters['Om0'],H0=self.parameters['H0'],
            w0=self.parameters['w0'],wa=self.parameters['wa'],astropy_conv=False)

        # Initialize the rate evolution/ z prior for a set of parameters
        dic_rate = {ll:self.parameters[ll] for ll in self.list_rate_param}
        zp_model = _redshift_prior(cosmo,name=self.rate_model,dic_param=dic_rate)

        # Update the sensitivity estimation with the new model
        self.injections.update_VT(mp_model,zp_model)

        # Below we calculate the likelihood as indicated in Eq. 7 on the tex document, see below for the terms
        Neff=self.injections.calculate_Neff()
        # If the injections are not enough return 0, you cannot go to that point
        if Neff<=(4*len(self.posterior_samples_dict)):
            return float(_np.nan_to_num(-_np.inf))

        # Below we calculate the likelihood as indicated in Eq. 7 on the tex document, see below for the terms
        if self.parallel:

            ms1, ms2, z_samples = _detector_frame_to_source_frame(
            cosmo,self.m1det_parallel,self.m2det_parallel,self.dl_parallel)

            log_new_prior_term = mp_model.log_joint_prob(ms1,ms2)+zp_model.log_prob(z_samples)
            # We remove the original prior here.
            log_jac_prior=_np.log(_np.abs(_detector_to_source_jacobian(z_samples, cosmo,dl=self.dl_parallel)))+2*_np.log(self.dl_parallel)

            # Selection effect, see Eq. 18 on paper
            beta = self.injections.gw_only_selection_effect()
            log_denominator = _np.log(beta)*len(self.posterior_samples_dict)

            #Note that here we have no normalization on the number of samples as they all have the same amount
            #Eq. 13 on the tex document (the numerator)
            log_single_ev_array=_logsumexp(log_new_prior_term-log_jac_prior,axis=1)-_np.log(log_new_prior_term.shape[1])

            # Calculate the numerator and denominator in Eq. 7 on the tex document  for each event and multiply them
            log_numerator = _np.sum(self.ln_evidences+log_single_ev_array)

            # Calulcate the poissonian term of Eq. 7 if the prior on rate is not log uiniform
            # Uses Eq. 18 on the paper basically
            if self.scale_free:
                log_poissonian_term = 0
            else:
                R0=self.parameters['R0']
                Nexp = self.injections.expected_number_detection(R0)
                log_poissonian_term = len(self.posterior_samples_dict)*_np.log(Nexp)-Nexp

            # Combine all the terms
            log_likeli = log_poissonian_term + log_numerator - log_denominator

            # Controls on the value of the log-likelihood. If the log-likelihood is -inf, then set it to the smallest
            # python valye 1e-309
            if log_likeli == _np.inf:
                raise ValueError('LOG-likelihood must be smaller than infinite')

            if _np.isnan(log_likeli):
                log_likeli = float(_np.nan_to_num(-_np.inf))
            else:
                log_likeli = float(_np.nan_to_num(log_likeli))

        # Same as above but done event by event with all the posterior samples (slower)
        else:
            log_likeli = _np.zeros(len(self.posterior_samples_dict))
            i = 0
            beta = self.injections.gw_only_selection_effect()
            log_denominator = _np.log(beta)

            for event in list(self.posterior_samples_dict.keys()):
                posterior_samples = self.posterior_samples_dict[event]
                ms1, ms2, z_samples = _detector_frame_to_source_frame(
                cosmo,posterior_samples.mass_1_det,posterior_samples.mass_2_det,posterior_samples.distance)

                log_new_prior_term = mp_model.log_joint_prob(ms1,ms2)+zp_model.log_prob(z_samples)
                # There is a normalization term there on dl.
                log_jac_prior =_np.log(_np.abs(_detector_to_source_jacobian(z_samples, cosmo,dl=posterior_samples.distance)))+2*_np.log(posterior_samples.distance)
                log_numerator = self.ln_evidences[i]+_logsumexp(log_new_prior_term-log_jac_prior)-_np.log(len(posterior_samples.mass_1_det))
                log_likeli[i] = log_numerator-log_denominator
                i+=1


            if self.scale_free:
                log_poissonian_term = 0
            else:
                R0=self.parameters['R0']
                Nexp = self.injections.expected_number_detection(R0)
                log_poissonian_term = len(self.posterior_samples_dict)*_np.log(Nexp)-Nexp

            log_likeli=_np.sum(log_likeli)+log_poissonian_term



            if log_likeli == _np.inf:
                raise ValueError('LOG-likelihood must be smaller than infinite')

            if _np.isnan(log_likeli):
                log_likeli = float(_np.nan_to_num(-_np.inf))
            else:
                log_likeli = float(_np.nan_to_num(log_likeli))

        return log_likeli

class hierarchical_analysis():
    """
    Class to manage and perform population analysis with a set of events.

    Parameters
    ----------
    pos_samples_dict: dic
        dictionary containing poster samples classes of the vents you want to analyse
    injections: injections class
        injections class to compute the selection effect from the injections module
    scale_free: boolean
        True if you want to marginalize on rate, False if not.
    """

    def __init__(self, pos_samples_dict, injections,scale_free=True):
        self.pos_samples_dict=pos_samples_dict
        self.injections=injections
        self.scale_free=scale_free

    def run_analysis_on_lists(self,list_pop_models,list_z_models, subset = None,z_em = None):
        """
        This method perform an hiererchical analysis over a list of cosmological, mass and redshift models.
        It returns a dictionary of single posterior (not normalized) for each event. Each posterior has on the x-axis
        the iteration over the list of models. Note that it assumes a log uniform prior on number of events (no rates or poissonian term)

        Parameters
        ----------
        list_cosmo_models: list
            List of cosmology classes to evaluate
        list_pop_models: list
            List of population models to evaluate
        list_z_models: list
            List of redshift models to evaluate
        subset: list
            Subset of events that you want to use
        z_em: dict 
            A list of detecte redshift from EM counterparts. Can be in the form of samples with uncertainties.
        """

        # Check if the population and mass models are of the same lenght
        check_condition = (len(list_pop_models)!=len(list_z_models))

        if check_condition:
            raise ValueError('The lenght of the models list must be the same')

        if subset is not None:
            analysis_events = subset
        else:
            analysis_events = list(self.pos_samples_dict.keys())

        numb_grid = len(list_pop_models)

        log_single_posterior = {}
        for event in analysis_events:
            log_single_posterior[event]=_np.ones(numb_grid)
         
        if z_em is not None:
            kde_distance = {}
            for event in analysis_events:
                kde_distance[event] = _gaussian_kde(self.pos_samples_dict[event].distance)

        # Run the analysis, calculate posterior basically, for each of the models provided event by event
        bar = progressbar.ProgressBar()
        for i in bar(range(numb_grid)):
            mp_model = list_pop_models[i]
            zp_model = list_z_models[i]
            cosmology = list_z_models[i].cosmo
            # Selection effect as in Eq 18 on the tex document
            self.injections.update_VT(mp_model,zp_model)
            beta = self.injections.gw_only_selection_effect()
            log_denominator = _np.log(beta)

            for event in analysis_events:
                # Numerator of the hierarchical likelihood as in Eq. 13 on the tex document
                posterior_samples = self.pos_samples_dict[event]
                ms1, ms2, z_samples = _detector_frame_to_source_frame(
                    cosmology,posterior_samples.mass_1_det,posterior_samples.mass_2_det,posterior_samples.distance)
                if z_em is None:
                    log_new_prior_term = mp_model.log_joint_prob(ms1,ms2)+zp_model.log_prob(z_samples)
                    log_jac_prior=_np.log(_np.abs(_detector_to_source_jacobian(z_samples, cosmology,dl=posterior_samples.distance)))+2*_np.log(posterior_samples.distance)
                    log_numerator = _logsumexp(log_new_prior_term-log_jac_prior)-_np.log(len(posterior_samples.mass_1_det))
                    log_denominator = _np.log(beta)
                    log_single_posterior[event][i] = log_numerator-log_denominator
                else:
                    if z_em[event] is None:
                        log_new_prior_term = mp_model.log_joint_prob(ms1,ms2)+zp_model.log_prob(z_samples)
                        log_jac_prior=_np.log(_np.abs(_detector_to_source_jacobian(z_samples, cosmology,dl=posterior_samples.distance)))+2*_np.log(posterior_samples.distance)
                        log_numerator = _logsumexp(log_new_prior_term-log_jac_prior)-_np.log(len(posterior_samples.mass_1_det))
                        log_denominator = _np.log(beta)
                        log_single_posterior[event][i] = log_numerator-log_denominator
                    else:             
                        mass_weight = _np.sum(mp_model.joint_prob(ms1,ms2))/len(ms1)
                        distance_weight = _np.sum(kde_distance[event](cosmology.dl_at_z(z_em[event]))*_np.power(cosmology.dl_at_z(z_em[event]),-2.)*zp_model.prob(z_em[event])*_np.power(1+z_em[event],-2.))/len(z_em[event])
                        log_single_posterior[event][i] = _np.log(distance_weight)+_np.log(mass_weight)-log_denominator
                        
                if _np.isnan(log_single_posterior[event][i]):
                    log_single_posterior[event][i] = float(_np.nan_to_num(-np.inf))
                else:
                    log_single_posterior[event][i] = float(_np.nan_to_num(log_single_posterior[event][i]))
                        
        return log_single_posterior

    def run_bilby(self, mass_model, cosmo_model, rate_model, prior_dict, parallel=10000,**kwargs):
        '''
        This method run Bilby to sample the hierarchical posteirior

        Parameters
        ----------
        mass_model: str
            The mass model used, either 'BBH-powerlaw', 'BBH-powerlaw-gaussian','BBH-broken-powerlaw'  and 'BBH-powerlaw-double-gaussian'
        cosmo_model: str
            The cosmological model used, either 'flatLCDM', 'w0flatLCDM' or 'w0waflatLCDM'
        rate_model: str
            The rate evolution model, either 'powerlaw' or 'madau'.
        parallel: integer or None
            If none, the likelihood is evaluated event by event using all posterior samples (slow method).
            If integer built a matrix of posterior samples = parallel for a quick computation.
        **kwargs: Bilby kwargs to run the sampler.

        Returns
        -------
        result: Bilby result object
        '''

        # Calls bilby routines to evaluate the prior.
        likeli = BBH_likelihood(self.pos_samples_dict,
        self.injections, mass_model = mass_model , cosmo_model = cosmo_model , rate_model=rate_model,
        parallel=parallel,scale_free=self.scale_free)
        result=_bilby.run_sampler(likelihood=likeli, priors=prior_dict,**kwargs)
        return result
