import numpy as _np
import astropy.units as _u
import astropy.constants as _c
from .conversion import *
from . import tcprior as _tcprior
import bilby as _bilby
from  scipy.special import beta as _beta
from scipy.interpolate import interp1d as _interp1d

class SineNdimensional(_bilby.prior.Prior):
    """
    Isotropic PDF for an angle theta between the axis 1 and axis i for a set of N axis
    The angles i1 for i not equal to N are defined between 0 and pi, while they are defined between
    0 and 2pi for N.

    Parameters
    ----------
    i: integer
        Axis w.r.t to the axis 1
    N: integer
        Total number of axis
    """
    def __init__(self,i,N):
        if i<2:
            raise ValueError('I should be greater than 2')
        elif i>N:
            raise ValueError('I should be less than N')
        if i==N:
            maxv=2*_np.pi
            minv=0.
        else:
            maxv=_np.pi
            minv=0.
        self.N=N
        self.i=i
        _bilby.prior.Prior.__init__(self, name='theta_i{:d}_1'.format(i), latex_label=r'$\theta_{{{:d},1}}$'.format(i),minimum=minv,maximum=maxv)

        if self.N==self.i:
            self.norm=_np.power(2.,1.)*_beta((self.N-self.i+1)/2.,1/2.)
        else:
            self.norm=_beta((self.N-self.i+1)/2.,1/2.)
        m_trial = _np.linspace(self.minimum,self.maximum,50000,endpoint=True)
        pdf_vals = self.prob(m_trial)
        cdf = _np.cumsum(pdf_vals)*(m_trial[1]-m_trial[0])
        cdf[0]=0
        cdf[-1]=1
        cdf[cdf>1]=1
        cdf[cdf<0]=0
        self.cdf_inverse = _interp1d(cdf,m_trial,bounds_error=True)

    def rescale(self, val):
        val=_np.reshape(val,-1)
        _bilby.prior.Prior.test_valid_for_rescaling(val)
        return self.cdf_inverse(val)

    def prob(self, val):
        val=_np.reshape(val,-1)
        toret=_np.power(_np.sin(val),self.N-self.i)/self.norm
        toret[(val<=self.minimum) | (val>=self.maximum)]=0.
        return toret


class absoluteSineNdimensional(_bilby.prior.Prior):
    """
    Isotropic PDF for an angle theta between the axis 1 and axis i for a set of N axis.
    This PDF only generates positive components for cartesian coordinates.
    The angles i1 for i not equal to N are defined between 0 and pi/2, while they are defined between
    0 and pi/2 for N.

    Parameters
    ----------
    i: integer
        Axis w.r.t to the axis 1
    N: integer
        Total number of axis
    """

    def __init__(self,i,N):

        if i<2:
            raise ValueError('I should be greater than 2')
        elif i>N:
            raise ValueError('I should be less than N')
        if i==N:
            maxv=_np.pi/2
            minv=0.
        else:
            maxv=_np.pi/2
            minv=0.

        self.N=N
        self.i=i

        _bilby.prior.Prior.__init__(self, name='theta_i{:d}_1'.format(i), latex_label=r'$\theta_{{{:d},1}}$'.format(i),minimum=minv,maximum=maxv)

        if self.N==self.i:
            self.norm=_np.power(2.,1.)*_beta((self.N-self.i+1)/2.,1/2.)/4
        else:
            self.norm=_beta((self.N-self.i+1)/2.,1/2.)/2

        m_trial = _np.linspace(self.minimum,self.maximum,100000,endpoint=True)
        pdf_vals = self.prob(m_trial)

        cdf = _np.cumsum(pdf_vals)*(m_trial[1]-m_trial[0])
        cdf[0]=0
        cdf[-1]=1
        cdf[cdf>1]=1
        cdf[cdf<0]=0
        self.cdf_inverse = _interp1d(cdf,m_trial,bounds_error=True)

    def rescale(self, val):
        val=_np.reshape(val,-1)
        _bilby.prior.Prior.test_valid_for_rescaling(val)
        return self.cdf_inverse(val)

    def prob(self, val):
        val=_np.reshape(val,-1)
        toret=_np.power(_np.sin(val),self.N-self.i)/self.norm
        toret[(val<=self.minimum) | (val>=self.maximum)]=0.
        return toret


def uniform_m1detm2det_in_m1q(m1,z):
    """
    Returns a uniform in detector frame masses prior written in terms of  m1,q

    Parameters
    ----------
    m1: np.array
        Source mass
    z: np.ndarray
        Redshift

    Returns
    -------
    Induced prior
    """
    return _np.power(z,2.)*m1

def dlsquare_in_redshift(dl,cosmology,z=None):
    """
    Returns the dl^2 prior written in redshift p(z)

    Parameters
    ----------
    dl: np.ndarray
        Luminosity distance in Mpc
    cosmology: astropy cosmology
        Astropy cosmology
    z: np.ndarray (optional)
        redshift correspoding to dl if none, it will be calculated

    Returns
    -------
    Induced prior
    """

    if z is None:
        z=_dl2z(dl,cosmology)
    dcom=cosmology.comoving_distance(z).to(_u.Mpc).value
    Hz=cosmology.H(z).value
    return _np.power(dl,2.)*(dcom+_c.c.to('km/s').value*(1+z)/Hz)

def uniform_aligned_spins_to_chieff(chieff,q,amax=1):
    """
    Returns uniform prion in spin magnitudes (aligned) in terms of chieff and q p(chi_eff|q).

    Parameters
    ----------
    chieff: np.ndarray
        Effective spin parameter
    q: np.array
        Mass ratio
    amax: float (optional)
        Maximum spin magnitde assumed by the original prior

    Returns
    -------
    Induced prior
    """
    return _tcprior.chi_effective_prior_from_aligned_spins(q,amax,chieff)

def uniform_isotropic_spins_to_chieff(chieff,q,amax=1):
    """
    Returns the uniform in spin magnitudes (isotropic) in terms of chieff and q p(chi_eff|q).

    Parameters
    ----------
    chieff: np.ndarray
        Effective spin parameter
    q: np.array
        Mass ratio
    amax: float (optional)
        Maximum spin magnitde assumed by the original prior

    Returns
    -------
    Induced prior
    """
    return _tcprior.chi_effective_prior_from_isotropic_spins(q,amax,chieff)

def uniform_isotropic_spins_to_chip(chip,q,amax=1):
    """
    Returns uniform prion in spin magnitudes (isotropic) in terms of chi_p and q p(chi_p|q).

    Parameters
    ----------
    chieff: np.ndarray
        Effective spin parameter
    q: np.array
        Mass ratio
    amax: float (optional)
        Maximum spin magnitde assumed by the original prior

    Returns
    -------
    Induced prior
    """
    return _tcprior.chi_p_prior_from_isotropic_spins(q,amax,chip)
