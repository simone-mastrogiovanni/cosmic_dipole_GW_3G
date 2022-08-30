import numpy as _np
from scipy.special import logsumexp as _logsumexp
from .utils.prior import absoluteSineNdimensional as _absoluteSineNdimensional
from .utils.conversion import ndangles2cartesian as _ndangles2cartesian
import bilby as _bilby
from tqdm import tqdm as _tqdm


class mixture_analyis(object):
    """
    This is the main class to handle a mixture analysis of synthtic populations

    Parameters
    ----------
    list_of_posterior_samples: obj
        List of posterior object
    list_of_syn_populations: obj
        The list of syntehtihc popultions that you would like to analyze, this should the object listofpop from the syntehtihc population module
    """
    def __init__(self,list_of_posterior_samples,list_of_syn_populations,montecarlo_mode):
        self.list_of_posterior_samples=list_of_posterior_samples
        self.list_of_syn_populations=list_of_syn_populations

        if (montecarlo_mode!='posterior') & (montecarlo_mode!='population'):
            raise ValueError('Montecarlo method not known')

        self.montecarlo_mode=montecarlo_mode

    def run_bilby(self,posterior_names=None,syn_names=None,prior_dict='isotropic',kernel_kwargs={},**kwargs):
        """
        Run bilby for the mixture model

        Parameters
        ----------
        posterior_names:  list (optional)
            list of string of the posteriors names you want to analyze if None we will analyze all events
        syn_names:  list (optional)
            list of string with the syntehtic models you want to analyze, if None we will analyze all syntetic models
        prior_dict: str or dict (optional)
            If string should be isotropic to indicate a prior isotropic in models, otherwise a dict of priors from Bilby. In this case
            the field name should be the name of the model
        kernel_kwargs: dict
            Dict containing the kwargs to pass to kernelize for the fit
        **kwargs: any kwargs to pass to bilby run_sampler

        Return
        ------
        res: Bilby result class
        """
        self.calculate_loglij_logsj_logexp_arrays(posterior_names=posterior_names,syn_names=syn_names,**kernel_kwargs)
        pmode=prior_dict
        if prior_dict=='isotropic':
            prior_dict={}
            Ntotal=len(self.syn_names)
            for i in _np.arange(2,Ntotal+1,1):
                pp=_absoluteSineNdimensional(i,Ntotal)
                prior_dict[pp.name]=pp

        res=_bilby.run_sampler(_mixture_bilby_likeli(self,prior_dict,prior_mode=pmode),prior_dict,**kwargs)
        return res

    def isotropic_to_fractions(self,res):
        """
        Converts posterior samples on isotropic angles to model fractions

        Parameters
        ----------
        res: Bilby result class
        Returns
        ------
        lambdas: dict
            Dictionary containing the fractions for each model
        """
        angs=list(res.posterior.keys())
        tot_samps=len(res.posterior[angs[0]])
        lambdas={self.syn_names[i]:_np.ones(tot_samps) for i in range(len(self.syn_names))}
        for i in range(tot_samps):
            angles=[]
            Ntotal=len(self.syn_names)
            angle=_np.array([res.posterior['theta_i{:d}_1'.format(j)][i] for j in _np.arange(2,Ntotal+1,1)])
            lamb=_ndangles2cartesian(angle)**2.
            for k in range(len(self.syn_names)):
                lambdas[self.syn_names[k]][i]=lamb[k]

        return lambdas

    def calculate_loglij_logsj_logexp(self,posterior_name,syn_name,kernel_mode='kde',**kwargs):
        """
        Calculates the Montecarlo integrals of the numerator, selection effect and number of expecting detections. Assumens that you have already kernelized
        The posterior or the synthetic population. Use calculate_loglij_logsj_logexp_arrays to kernelize authomatically
        Returning their log

        Parameters
        ----------
        posterior_name:  string
            Name of posterior sample to analyze
        syn_name:  str
            Name of model to analyze
        kernel_mode: str (optional)
            How to fit the posterior samples, either histogram or kde.
        **kwargs: Any argument to pass to the kernel fitting (gaussian_kde or ndhistogram)

        Return
        ------
        log of the Likelihood integrated, log of the selection effect, log of the expected number of detections
        """

        if self.montecarlo_mode=='population':
            loglij=self.list_of_posterior_samples.posterior_list[posterior_name].calculate_model_match(self.list_of_syn_populations.population_list[syn_name])
        elif self.montecarlo_mode=='posterior':
            loglij=self.list_of_syn_populations.population_list[syn_name].calculate_model_match(self.list_of_posterior_samples.posterior_list[posterior_name])

        logsj=_np.log(self.list_of_syn_populations.population_list[syn_name].selection_effect)
        logexp=_np.log(self.list_of_syn_populations.population_list[syn_name].expected_detections)
        logNtot=_np.log(self.list_of_syn_populations.population_list[syn_name].Ntotal)

        return loglij,logsj,logexp,logNtot

    def calculate_loglij_logsj_logexp_arrays(self,posterior_names=None,syn_names=None,kernel_mode='kde',**kwargs):
        """
        Initialize the integrated likelihood in a matrix i-events X j-models, the selection effect and the number of expected
        detections in arrays with len j-models

        Parameters
        ----------
        posterior_names:  list (optional)
            list of string of the posteriors names you want to analyze if None we will analyze all events
        syn_names:  list (optional)
            list of string with the syntehtic models you want to analyze, if None we will analyze all syntetic models
        kernel_kwargs: dict
            Dict containing the kwargs to pass to kernelize for the fit
        **kwargs: Any argument to pass to the kernel fitting (gaussian_kde or ndhistogram)
        """

        if posterior_names is None:
            posterior_names=list(self.list_of_posterior_samples.posterior_list.keys())
            self.posterior_names=posterior_names
        else:
            self.posterior_names=posterior_names
        if syn_names is None:
            syn_names=list(self.list_of_syn_populations.population_list.keys())
            self.syn_names=syn_names
        else:
            self.syn_names=syn_names

        self.loglij=_np.zeros((len(posterior_names),len(syn_names)))
        self.logsj=_np.zeros(len(syn_names))
        self.logexpj=_np.zeros(len(syn_names))
        self.logNtot=_np.zeros(len(syn_names))
        #self.logpriors=_np.zeros(len(syn_names))

        if self.montecarlo_mode=='posterior':
            for j,key_j in _tqdm(enumerate(syn_names),desc='Kernelizing populations'):
                self.list_of_syn_populations.population_list[key_j].kernelize(kernel_mode=kernel_mode,**kwargs)
        elif self.montecarlo_mode=='population':
            for i,key_i in _tqdm(enumerate(posterior_names),desc='Kernelizing posteriors'):
                self.list_of_posterior_samples.posterior_list[key_i].kernelize(kernel_mode=kernel_mode,**kwargs)

        for i,key_i in _tqdm(enumerate(posterior_names),desc='Calculating match for Posterior #'):
            for j,key_j in enumerate(syn_names):
                self.loglij[i,j],self.logsj[j],self.logexpj[j],self.logNtot[j]=self.calculate_loglij_logsj_logexp(key_i,key_j,kernel_mode=kernel_mode,**kwargs)
                #self.logpriors[j]=_np.log(self.list_of_syn_populations.population_list[key_j].prior_model)

        for i,key_i in enumerate(posterior_names):
            if _np.isinf(self.loglij[i,:]).all():
                raise ValueError('The GW event {:s} has 0 likelihood for all the models, please check the histograms or KDE fitting of the event or your models'.format(key_i))

    def calculate_log_hierachical_likelihood(self,lambdasquare_dict):
        """
        Calculate the hierarchical likelihood given a set of mixtures

        Parameters
        ----------
        lambdasquare_dict:  dict
            Dictionary containing the mixing coefficients for each model. The field name of the dict must be the name of the model.
            Note that if isotrpic prior on model is chosen, lambdassquare should sum to one, otherwise no.
        prior_mode:  string or dict
            If 'isotropic' we will assume isotropic model mixin (sum to 1) otherwise (if dict) we will assume that the models does not sum to unity

        Returns
        -------
        hierarchical log likelihood.
        """


        lambdasquare=_np.array([lambdasquare_dict[key] for key in self.syn_names])

        Nobs=self.loglij.shape[0]
        #new_log_priors=self.logpriors+_np.log(lambdasquare)-_logsumexp(self.logpriors+_np.log(lambdasquare))
        new_log_priors=_np.log(lambdasquare)+self.logNtot-_logsumexp(_np.log(lambdasquare)+self.logNtot)
        #print('Ciao',_np.exp(new_log_priors))
        new_log_priors[_np.log(lambdasquare)==-_np.inf]=-_np.inf
        new_log_priors_matrix=_np.tile(new_log_priors,(self.loglij.shape[0],1))

        # if prior_mode=='isotropic':
        #     Nexp=_np.exp(self.logexpj+new_log_priors)
        # else:
        #     logpnew=_logsumexp(_np.log(lambdasquare))+self.logpriors+_np.log(lambdasquare)-_logsumexp(self.logpriors+_np.log(lambdasquare))
        #     logpnew[_np.log(lambdasquare)==-_np.inf]=-_np.inf
        #     Nexp=_np.exp(self.logexpj+logpnew)

        logNexp=self.logexpj+_np.log(lambdasquare)
        logNexp=_logsumexp(logNexp)
        toret=logNexp*Nobs-_np.exp(logNexp)+_np.sum(_logsumexp(self.loglij+new_log_priors_matrix,axis=1)-_logsumexp(self.logsj+new_log_priors))

        return toret



class _mixture_bilby_likeli(_bilby.Likelihood):
    """
    This is an internal class to handle the hierarchical log likelihood in bilby

    Parameters
    ----------

    mixture_analyis: obj
        mixture analysis object from this module
    prior_mode:  string or dict
        If 'isotropic' we will assume isotropic model mixin (sum to 1) otherwise (if dict) we will assume that the models does not sum to unity
    """
    def __init__(self,mixture_analyis,prior_dict,prior_mode='isotropic'):
        # Initialize the Bilby parameters
        self.prior_mode=prior_mode
        self.mixture_analyis=mixture_analyis
        super().__init__(parameters={})

    def log_likelihood(self):

        if self.prior_mode=='isotropic':
            angles=[]
            Ntotal=len(self.mixture_analyis.syn_names)
            angle=_np.array([self.parameters['theta_i{:d}_1'.format(i)] for i in _np.arange(2,Ntotal+1,1)])
            lamb=_ndangles2cartesian(angle)**2.
            lambdas={self.mixture_analyis.syn_names[i]:lamb[i] for i in range(len(self.mixture_analyis.syn_names))}
        else:
            lambdas={key: self.parameters[key] for key in self.parameters.keys()}

        return self.mixture_analyis.calculate_log_hierachical_likelihood(lambdas)
