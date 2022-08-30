from scipy.interpolate import interp1d as _interp1d
import numpy as _np
from astropy import units as _u
import copy as _copy
from scipy.integrate import quad as _quad
from scipy.integrate import cumtrapz as _cumtrapz
from scipy.integrate import simps as _simps

class log_powerlaw_rate(object):
    def __init__(self,gamma):
        self.gamma=gamma
    def __call__(self,z):
        return self.gamma*_np.log1p(z)

class log_madau_rate(object):
    def __init__(self,gamma,kappa,zp):
        self.gamma=gamma
        self.kappa=kappa
        self.zp=zp
    def __call__(self,z):
        return _np.log1p(_np.power(1+self.zp,-self.gamma-self.kappa))+self.gamma*_np.log1p(z)-_np.log1p(_np.power((1+z)/(1+self.zp),self.gamma+self.kappa))

def log1xspace(minv,maxv,nsteps):
    return _np.expm1(_np.linspace(_np.log1p(minv),_np.log1p(maxv),nsteps))

class redshift_prior(object):
    """
    This class handles the redshift prior. The redshift prior is properly normalized and it is
    :math:`p(z) \\propto \\frac{(1+z)^\\gamma}{(1+z)} \\frac{dV_c}{dz} \\psi(z|z_p,k,\\gamma,k)`
    where
    :math:`p(z) \\frac{\\psi(z|z_p,k,\\gamma,k)=(1+z_p)^{-\\lambda-k}}{1+\\left(\\frac{1+z}{1+z_p} \\right)^{k+\\lambda}}`

    Parameters
    ----------
    cosmo: cosmology class
        Cosmology class from its module
    name: str
        'powerlaw','madau'
    dic_param: dic
        Dictiornary containing the parameters of the model
    """
    def __init__(self,cosmo,name,dic_param):

        self.name=name
        self.dic_param=_copy.deepcopy(dic_param)
        self.cosmo = _copy.deepcopy(cosmo)

        # Define the comoving volume time in Gpc^3

        if self.name=='powerlaw':
            # Build a powerlaw evolution model for the rates, Eq. E 16 of https://arxiv.org/pdf/2010.14533.pdf
            gamma = dic_param['gamma']
            self.log_rate_eval = log_powerlaw_rate(gamma)
        elif self.name=='madau':
            # Build a madau-like model for the rates Eq. 2 of https://arxiv.org/abs/2003.12152
            gamma = dic_param['gamma']
            zp = dic_param['zp']
            kappa = dic_param['kappa']

            self.log_rate_eval = log_madau_rate(gamma,kappa,zp)
        else:
            raise ValueError('Z-rate prior not known')

        z_trial=log1xspace(0.,cosmo.zmax,5000)
        prior_trial = _np.exp(_np.log(cosmo.dVc_by_dz(z_trial))-_np.log1p(z_trial)+self.log_rate_eval(z_trial))
        # The norm fact is defined in such a way that when applied it gives a probabity in redshift
        # When you do not apply it you have something in Gpc^3
        self.norm_fact = _simps(prior_trial,z_trial)

    def log_prob(self,z_vals):
        """
        Returns the probability

        Parameters
        ----------
        z_vals: np.array
            Redshift values at which to compute the probability
        astro_norm: bool
            If True returns the prior not normalized, that you can multiply for rates
            basically :math:`p(z) = \\frac{R(z)}{R_0}\\frac{1}{(1+z)} \\frac{dV_c}{dz}`
        """
        return _np.log(self.cosmo.dVc_by_dz(z_vals))-_np.log1p(z_vals)+self.log_rate_eval(z_vals)-_np.log(self.norm_fact)

    def prob(self,z_vals):
        """
        Returns the probability

        Parameters
        ----------
        z_vals: np.array
            Redshift values at which to compute the probability
        astro_norm: bool
            If True returns the prior not normalized, that you can multiply for rates
            basically :math:`p(z) = \\frac{R(z)}{R_0}\\frac{1}{(1+z)} \\frac{dV_c}{dz}`
        """
        return _np.exp(self.log_prob(z_vals))

    def log_prob_astro(self,z_vals):
        """
        Returns the probability

        Parameters
        ----------
        z_vals: np.array
            Redshift values at which to compute the probability
        astro_norm: bool
            If True returns the prior not normalized, that you can multiply for rates
            basically :math:`p(z) = \\frac{R(z)}{R_0}\\frac{1}{(1+z)} \\frac{dV_c}{dz}`
        """
        return _np.log(self.cosmo.dVc_by_dz(z_vals))-_np.log1p(z_vals)+self.log_rate_eval(z_vals)

    def prob_astro(self,z_vals):
        """
        Returns the probability

        Parameters
        ----------
        z_vals: np.array
            Redshift values at which to compute the probability
        astro_norm: bool
            If True returns the prior not normalized, that you can multiply for rates
            basically :math:`p(z) = \\frac{R(z)}{R_0}\\frac{1}{(1+z)} \\frac{dV_c}{dz}`
        """
        return _np.exp(self.log_prob_astro(z_vals))


    def sample(self, Nsample):
        """
        This method samples from the joint probability :math:`p(z)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        """

        vals_z= _np.random.rand(Nsample)
        z_trials = log1xspace(0,self.cosmo.zmax,50000)

        cumulative_disc = _cumtrapz(self.prob(z_trials),z_trials)
        # The CDF is 0 below the minimum and 1 above the maximum
        cumulative_disc[0]=0
        cumulative_disc[-1]=1
        cdf=_interp1d(z_trials[:-1:],cumulative_disc,bounds_error=False,fill_value=(0,1))

        cdf_z_trials = cdf(z_trials[:-1:])

        interpo_icdf_z = _interp1d(cdf_z_trials,z_trials[:-1:],kind='cubic')
        z_samples = interpo_icdf_z(vals_z)

        return z_samples
