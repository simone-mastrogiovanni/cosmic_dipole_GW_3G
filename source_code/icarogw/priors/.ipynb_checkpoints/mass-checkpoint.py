import numpy as _np
import copy as _copy
import sys as _sys
from .custom_bilby_priors import DoubleTaperedPowerLaw as _DoubleTaperedPowerLaw
from .custom_bilby_priors import TaperedPowerLawGaussian as _TaperedPowerLawGaussian
from bilby.core.prior import  PowerLaw as _PowerLaw
from bilby.core.prior import ConditionalPowerLaw as _ConditionalPowerLaw
from bilby.core.prior import ConditionalPriorDict as _ConditionalPriorDict
from . import custom_math_priors as _cmp
from scipy.interpolate import interp1d as _interp1d

def condition_func_m1m2(reference_params, mass_1):
    import bilby
    return dict(minimum=reference_params['minimum'], maximum=mass_1)

class mass_prior(object):
    """
    Wrapper for for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`

    Parameters
    ----------
    name: str
        Name of the model that you want. Available 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw', 'BBH-powerlaw-double-gaussian'.
    hyper_params_dict: dict
        Dictionary of hyperparameters for the prior model you want to use. See code lines for more details
    bilby_priors: boolean
        If you want to use bilby priors or not. It is faster to use the analytical functions.
    """

    def __init__(self, name, hyper_params_dict):

        self.name = name
        self.hyper_params_dict=_copy.deepcopy(hyper_params_dict)
        dist = {}

        if self.name == 'BBH-powerlaw':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']

            # Define the prior on the masses with a truncated powerlaw as in Eq.33,34,35 on the tex document
            dist={'mass_1':_cmp.PowerLaw_math(alpha=-alpha,min_pl=mmin,max_pl=mmax),
            'mass_2':_cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=mmax)}

            self.mmin = mmin
            self.mmax = mmax

        elif self.name == 'BBH-powerlaw-gaussian':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']

            mu_g = hyper_params_dict['mu_g']
            sigma_g = hyper_params_dict['sigma_g']
            lambda_peak = hyper_params_dict['lambda_peak']

            delta_m = hyper_params_dict['delta_m']

            # Define the prior on the masses with a truncated powerlaw + gaussian
            # as in Eq.36-37-38 on the tex document
            m1pr = _cmp.PowerLawGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_peak
            ,mean_g=mu_g,sigma_g=sigma_g,min_g=mmin,max_g=mu_g+5*sigma_g)
            m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=m1pr.maximum)

            # Smooth the lower end of these distributions
            dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
            'mass_2':_cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}

            # TODO Assume that the gaussian peak does not overlap too much with the mmin
            self.mmin = mmin
            self.mmax = dist['mass_1'].maximum

        elif self.name == 'BBH-powerlaw-double-gaussian':

            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']

            mu_g_low = hyper_params_dict['mu_g_low']
            sigma_g_low = hyper_params_dict['sigma_g_low']
            mu_g_high = hyper_params_dict['mu_g_high']
            sigma_g_high = hyper_params_dict['sigma_g_high']

            lambda_g = hyper_params_dict['lambda_g']
            lambda_g_low = hyper_params_dict['lambda_g_low']

            delta_m = hyper_params_dict['delta_m']

            # Define the prior on the masses with a truncated powerlaw + gaussian
            # as in Eq.45-46-448 on the tex document
            m1pr = _cmp.PowerLawDoubleGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_g, lambda_g_low=lambda_g_low
            ,mean_g_low=mu_g_low,sigma_g_low=sigma_g_low,mean_g_high=mu_g_high,sigma_g_high=sigma_g_high
            , min_g=mmin,max_g=mu_g_high+5*_np.max([sigma_g_low,sigma_g_high]))
            m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=m1pr.maximum)

            # Smooth the lower end of these distributions
            dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
            'mass_2':_cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}

            # TODO Assume that the gaussian peak does not overlap too much with the mmin
            self.mmin = mmin
            self.mmax = dist['mass_1'].maximum

        elif self.name == 'BBH-broken-powerlaw':
            alpha_1 = hyper_params_dict['alpha_1']
            alpha_2 = hyper_params_dict['alpha_2']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']
            b =  hyper_params_dict['b']

            delta_m = hyper_params_dict['delta_m']

            # Define the prior on the masses with a truncated powerlaw + gaussian
            # as in Eq.39-42-43 on the tex document
            m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-alpha_1,alpha_2=-alpha_2,min_pl=mmin,max_pl=mmax,b=b)
            m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=mmax)

            # Smooth the lower end of these distributions
            dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
            'mass_2':_cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}

            self.mmin = mmin
            self.mmax = mmax

        else:
            print('Name not known, aborting')
            _sys.exit()

        self.dist = dist

    def joint_prob(self, ms1, ms2):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        ms1: np.array(matrix)
            mass one in solar masses
        ms2: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper

        return _np.exp(self.log_joint_prob(ms1,ms2))

    def log_joint_prob(self, ms1, ms2):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        ms1: np.array(matrix)
            mass one in solar masses
        ms2: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = self.dist['mass_1'].log_prob(ms1)+self.dist['mass_2'].log_conditioned_prob(ms2,self.mmin*_np.ones_like(ms1),ms1)
        to_ret[_np.isnan(to_ret)]=-_np.inf

        return to_ret

    def sample(self, Nsample):
        """
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        """

        vals_m1 = _np.random.rand(Nsample)
        vals_m2 = _np.random.rand(Nsample)

        m1_trials = _np.logspace(_np.log10(self.dist['mass_1'].minimum),_np.log10(self.dist['mass_1'].maximum),50000)
        m2_trials = _np.logspace(_np.log10(self.dist['mass_2'].minimum),_np.log10(self.dist['mass_2'].maximum),50000)

        cdf_m1_trials = self.dist['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.dist['mass_2'].cdf(m2_trials)

        m1_trials = _np.log10(m1_trials)
        m2_trials = _np.log10(m2_trials)

        _,indxm1 = _np.unique(cdf_m1_trials,return_index=True)
        _,indxm2 = _np.unique(cdf_m2_trials,return_index=True)

        interpo_icdf_m1 = _interp1d(cdf_m1_trials[indxm1],m1_trials[indxm1],bounds_error=False,fill_value=(m1_trials[0],m1_trials[-1]))
        interpo_icdf_m2 = _interp1d(cdf_m2_trials[indxm2],m2_trials[indxm2],bounds_error=False,fill_value=(m2_trials[0],m2_trials[-1]))

        mass_1_samples = 10**interpo_icdf_m1(vals_m1)
        mass_2_samples = 10**interpo_icdf_m2(vals_m2*self.dist['mass_2'].cdf(mass_1_samples))

        to_ret = {'mass_1':mass_1_samples,'mass_2':mass_2_samples}

        return to_ret['mass_1'], to_ret['mass_2']
