import bilby as _bilby
import numpy as _np
from scipy.interpolate import interp1d as _interp1d

__all__ = ['S_factor','PowerLawGaussian','DoubleTaperedPowerLaw',
'TaperedPowerLaw','TaperedPowerLawGaussian']

def S_factor(mass, mmin,delta_m):

    if not isinstance(mass,_np.ndarray):
        mass = _np.array([mass])

    to_ret = _np.ones_like(mass)
    if delta_m == 0:
        return to_ret

    mprime = mass-mmin

    select_window = (mass>mmin) & (mass<(delta_m+mmin))
    select_one = mass>=(delta_m+mmin)
    select_zero = mass<=mmin

    effe_prime = _np.ones_like(mass)
    effe_prime[select_window] = _np.exp(_np.nan_to_num((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
    to_ret = 1./(effe_prime+1)
    to_ret[select_zero]=0.
    to_ret[select_one]=1.
    return to_ret


class PowerLawGaussian(_bilby.prior.Prior):
    """This is a powerlaw + gaussian"""

    def __init__(self, alpha_pl, min_pl,max_pl, mu_g, sigma_g, lambda_peak, name=None, latex_label=None, global_minimum=None, global_maximum=None):

        # TODO: Add a control on the global minimum and global maximum when provided
        if global_minimum is None :
            global_minimum = _np.min([min_pl,mu_g-5*sigma_g])
        if global_maximum is None :
            global_maximum = _np.max([max_pl,mu_g+5*sigma_g])

        _bilby.prior.Prior.__init__(self, name=name, latex_label=latex_label,minimum=global_minimum,maximum=global_maximum)

        self.alpha_pl = alpha_pl
        self.mu_g = mu_g
        self.sigma_g = sigma_g
        self.lambda_peak = lambda_peak
        self.min_pl = min_pl
        self.max_pl = max_pl

        self.pplow = _bilby.core.prior.PowerLaw(alpha=self.alpha_pl,minimum=min_pl,maximum=max_pl)
        self.gpeak = _bilby.core.prior.TruncatedGaussian(mu=self.mu_g,sigma=self.sigma_g,
        minimum=global_minimum,maximum=global_maximum)

        m_trial = _np.linspace(global_minimum,global_maximum,20000,endpoint=True)
        pdf_vals = self.prob(m_trial)
        cdf = _np.cumsum(pdf_vals)*(m_trial[1]-m_trial[0])
        cdf[0]=0
        cdf[-1]=1
        cdf[cdf>1]=1
        cdf[cdf<0]=0
        self.cdf_inverse = _interp1d(cdf,m_trial,bounds_error=True)

    def rescale(self, val):
        _bilby.prior.Prior.test_valid_for_rescaling(val)
        return self.cdf_inverse(val)

    def prob(self, val):
        toret = self.pplow.prob(val)*(1-self.lambda_peak)+self.lambda_peak*self.gpeak.prob(val)
        return toret

class DoubleTaperedPowerLaw(_bilby.prior.Prior):
    """This is the tapered powerlaw"""

    def __init__(self, b, alpha_1, alpha_2, minimum, maximum, delta_m, name=None, latex_label=None):
        # Used to override the maximum
        _bilby.prior.Prior.__init__(self, name=name, latex_label=latex_label,minimum=minimum,maximum=maximum)
        self.delta_m = delta_m
        self.alpha_1=alpha_1
        self.alpha_2=alpha_2
        self.b=b
        self.break_point = self.minimum+b*(self.maximum-self.minimum)

        self.pplow_1 = TaperedPowerLaw(alpha_pl=self.alpha_1,minimum=minimum,maximum=maximum,delta_m=delta_m)
        self.pplow_2 = TaperedPowerLaw(alpha_pl=self.alpha_2,minimum=minimum,maximum=maximum,delta_m=delta_m)

        m_trial = _np.linspace(minimum,maximum,20000,endpoint=True)
        pdf_vals = self.prob(m_trial)
        cdf = _np.cumsum(pdf_vals)*(m_trial[1]-m_trial[0])
        cdf[0]=0
        cdf[-1]=1
        cdf[cdf>1]=1
        cdf[cdf<0]=0
        self.cdf_inverse = _interp1d(cdf,m_trial,bounds_error=True)

    def rescale(self, val):
        _bilby.prior.Prior.test_valid_for_rescaling(val)
        return self.cdf_inverse(val)

    def prob(self, val):

        if not hasattr(self, 'correction'):
            m_trial = _np.linspace(self.minimum,self.maximum,20000,endpoint=True)
            # 1.0001 done for numerical issues on the edge
            self.pplow2_w_scale = self.pplow_2.prob(self.break_point)/self.pplow_1.prob(self.break_point)
            part_1 = self.pplow_1.prob(m_trial)
            part_2 = self.pplow_2.prob(m_trial)/self.pplow2_w_scale
            part_1[m_trial>self.break_point]=0.
            part_2[m_trial<=self.break_point]=0.
            toret = part_1+part_2
            self.correction = _np.trapz(toret,m_trial)

        part_1 = self.pplow_1.prob(val)
        part_2 = self.pplow_2.prob(val)/self.pplow2_w_scale
        part_1[val>self.break_point]=0.
        part_2[val<=self.break_point]=0.
        toret = (part_1+part_2)/self.correction
        return toret

class TaperedPowerLaw(_bilby.prior.Prior):
    """This is the tapered powerlaw"""

    def __init__(self, alpha_pl, minimum, maximum, delta_m, name=None, latex_label=None):
        # Used to override the maximum
        _bilby.prior.Prior.__init__(self, name=name, latex_label=latex_label,minimum=minimum,maximum=maximum)
        self.delta_m = delta_m
        self.alpha_pl=alpha_pl

        self.pplow = _bilby.core.prior.PowerLaw(alpha=self.alpha_pl,minimum=minimum,maximum=maximum)

        m_trial = _np.linspace(minimum,maximum,20000,endpoint=True)
        pdf_vals = self.prob(m_trial)
        cdf = _np.cumsum(pdf_vals)*(m_trial[1]-m_trial[0])
        cdf[0]=0
        cdf[-1]=1
        cdf[cdf>1]=1
        cdf[cdf<0]=0
        self.cdf_inverse = _interp1d(cdf,m_trial,bounds_error=True)

    def rescale(self, val):
        _bilby.prior.Prior.test_valid_for_rescaling(val)
        return self.cdf_inverse(val)

    def prob(self, val):

        if not hasattr(self, 'correction'):
            m_trial = _np.linspace(self.minimum,self.maximum,20000,endpoint=True)
            toret = self.pplow.prob(m_trial)*S_factor(m_trial,self.minimum,self.delta_m)
            self.correction = _np.trapz(toret,m_trial)

        toret = self.pplow.prob(val)*S_factor(val,self.minimum,self.delta_m)/self.correction
        return toret

class TaperedPowerLawGaussian(_bilby.prior.Prior):
    """This is a powerlaw + gaussian"""

    def __init__(self, alpha_pl, min_pl, max_pl, mu_g, sigma_g, lambda_peak,delta_m, name=None, latex_label=None,global_minimum=None, global_maximum=None):
        # TODO: Add a control on the global minimum and global maximum when provided
        if global_minimum is None :
            global_minimum = _np.min([min_pl,mu_g-5*sigma_g])
        if global_maximum is None :
            global_maximum = _np.max([max_pl,mu_g+5*sigma_g])

        _bilby.prior.Prior.__init__(self, name=name, latex_label=latex_label,minimum=global_minimum,maximum=global_maximum)


        self.alpha_pl = alpha_pl
        self.mu_g = mu_g
        self.sigma_g = sigma_g
        self.lambda_peak = lambda_peak
        self.min_pl = min_pl
        self.max_pl = max_pl
        self.delta_m = delta_m

        self.pplow = _bilby.core.prior.PowerLaw(alpha=self.alpha_pl,minimum=min_pl,maximum=max_pl)
        self.gpeak = _bilby.core.prior.TruncatedGaussian(mu=self.mu_g,sigma=self.sigma_g,
        minimum=global_minimum,maximum=global_maximum)

        m_trial = _np.linspace(global_minimum,global_maximum,20000,endpoint=True)
        pdf_vals = self.prob(m_trial)
        cdf = _np.cumsum(pdf_vals)*(m_trial[1]-m_trial[0])
        cdf[0]=0
        cdf[-1]=1
        cdf[cdf>1]=1
        cdf[cdf<0]=0
        self.cdf_inverse = _interp1d(cdf,m_trial,bounds_error=True)

    def rescale(self, val):
        _bilby.prior.Prior.test_valid_for_rescaling(val)
        return self.cdf_inverse(val)

    def prob(self, val):

        if not hasattr(self, 'correction'):
            m_trial = _np.linspace(self.minimum,self.maximum,20000,endpoint=True)
            toret = (self.pplow.prob(m_trial)*(1-self.lambda_peak)+self.lambda_peak*self.gpeak.prob(m_trial))*S_factor(m_trial,self.minimum,self.delta_m)
            self.correction = _np.trapz(toret,m_trial)

        toret = (self.pplow.prob(val)*(1-self.lambda_peak)+self.lambda_peak*self.gpeak.prob(val))*S_factor(val,self.minimum,self.delta_m)/self.correction

        return toret
