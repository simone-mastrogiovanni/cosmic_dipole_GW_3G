"""
Module for managing cosmology with classes
"""

import numpy as _np
from astropy.cosmology import FlatLambdaCDM as _FlatLambdaCDM
from astropy.cosmology import FlatwCDM as _FlatwCDM
from astropy.cosmology import Flatw0waCDM as _Flatw0waCDM

from astropy.cosmology import z_at_value as _z_at_value

from astropy import constants as _constants
from astropy import units as _u
from scipy.interpolate import interp1d as _interp1d
from scipy.interpolate import splev as _splev
from scipy.interpolate import splrep as _splrep


__all__= ['flatLCDM','w0flatLCDM','w0waflatLCDM']

def log1xspace(min,max,nsteps):
    return _np.expm1(_np.linspace(_np.log1p(min),_np.log1p(max),nsteps))

class flatLCDM(object):
    """
    A Class to work with flat LCDM cosmology
    """

    def __init__(self, Omega_m=0.308,H0=67,zmax=10, astropy_conv=False):
        """ Initialize the cosmology

        Parameters
        ----------
        Omega_m : float
            Fraction of matter energy
        H0 : float
            Hubble constant today in km/Mpc/s
        zmax : float
            Maximum redshift used for the cosmology
        astropy_conv : boolean
            Wether luminosity distance - redshift conversions are done with astropy
        """

        self.model_params = ['Om0','H0']
        self.astropy_cosmology = _FlatLambdaCDM(Om0=Omega_m,H0=H0)
        self.zmax=zmax
        self.astropy_conv=astropy_conv

        # Calculate a lookup tables for the: luminosity distance, redshift and differential of comoving volume
        z_array = _np.logspace(-4,_np.log10(self.zmax),  2500)
        dl_trials = _np.log10(self.astropy_cosmology.luminosity_distance(z_array).value)
        dvc_dz_trials =  _np.log10(4*_np.pi*self.astropy_cosmology.differential_comoving_volume(z_array).to(_u.Gpc**3/_u.sr).value)

        z_array = _np.log10(z_array)

        # Interpolate the lookup tables
        self.interp_dvc_dz = _splrep(z_array,dvc_dz_trials)
        self.interp_dl_to_z = _splrep(dl_trials,z_array)
        self.interp_z_to_dl = _splrep(z_array,dl_trials)

    def dl_at_z(self, z):
        """
        Returns luminosity distance in Mpc given distance and cosmological parameters

        Parameters
        ----------
        z : np.array (or matrix)
            Cosmological redshift
        H0 : float
            Hubble constant today in km/Mpc/s
        """

        return _np.nan_to_num(10.**_splev(_np.log10(z),self.interp_z_to_dl,ext=0))

    def z_at_dl(self,dl):
        """
        Given a luminosity distance, this method returns
        the redshift

        Parameters
        ----------
        dl: _np.array
            Luminosity distance in Mpc
        """

        if not isinstance(dl,_np.ndarray):
            dl = _np.array([dl])

        if self.astropy_conv:
            z_ret = _np.array([_z_at_value(self.astropy_cosmology.luminosity_distance,
                d*_u.Mpc,zmax=self.zmax) for d in dl])
        else:
            z_ret = _np.nan_to_num(10.**_splev(_np.log10(dl),self.interp_dl_to_z,ext=0))
        return z_ret

    def dVc_by_dz(self,z):
        """
        Returns the differential in comoving volume in Units of :math:`{\\rm Gpc}^3`

        Parameters
        ----------
        z: _np.array
            Redshift

        """

        return _np.nan_to_num(10.**_splev(_np.log10(z),self.interp_dvc_dz,ext=0))

    def dL_by_dz(self, z,dl=None):
        """
        Calculates the d_dL/dz for this cosmology

        Parameters
        ----------
        z: _np. arrays
            Redshift
        dl: _np.arrays
            optional value of dl to speed up the code
        """
        speed_of_light = _constants.c.to('km/s').value
        # Calculate the Jacobian of the luminosity distance w.r.t redshift
        if dl is None:
            return self.dl_at_z(z)/(1+z) + speed_of_light*(1+z)/(self.astropy_cosmology.H0.value*self.Efunc(z))
        else:
            return dl/(1+z) + speed_of_light*(1+z)/(self.astropy_cosmology.H0.value*self.Efunc(z))


    def Efunc(self,z):
        """
        Returns the :math:`E(z)=\\sqrt{\\Omega_{m,0}(1+z)^3+\\Omega_{\\Lambda}}` function for this cosmology

        Parameters
        ----------
        z: _np.array
            Redshift
        """

        return _np.sqrt(self.astropy_cosmology.Om0*_np.power(1+z,3)+(1-self.astropy_cosmology.Om0))




class w0flatLCDM(object):
    """
    A Class to work with flat w0LCDM cosmology
    """

    def __init__(self, Omega_m=0.308,H0=67,w0=-1.0,zmax=10, astropy_conv=False):
        """ Initialize the cosmology

        Parameters
        ----------
        Omega_m : float
            Fraction of matter energy
        H0 : float
            Hubble constant today in km/Mpc/s
        w0 : float
            Dark energy EOS parameter
        zmax : float
            Maximum redshift used for the cosmology
        astropy_conv : boolean
            Wether luminosity distance - redshift conversions are done with astropy
        """

        self.model_params = ['Om0','H0','w0']
        self.astropy_cosmology = _FlatwCDM(Om0=Omega_m,H0=H0,w0=w0)
        self.zmax=zmax
        self.astropy_conv=astropy_conv

        # Calculate a lookup tables for the: luminosity distance, redshift and differential of comoving volume
        z_array = _np.logspace(-4,_np.log10(self.zmax),  2500)
        dl_trials = _np.log10(self.astropy_cosmology.luminosity_distance(z_array).value)
        dvc_dz_trials =  _np.log10(4*_np.pi*self.astropy_cosmology.differential_comoving_volume(z_array).to(_u.Gpc**3/_u.sr).value)

        z_array = _np.log10(z_array)

        # Interpolate the lookup tables
        self.interp_dvc_dz = _splrep(z_array,dvc_dz_trials)
        self.interp_dl_to_z = _splrep(dl_trials,z_array)
        self.interp_z_to_dl = _splrep(z_array,dl_trials)

    def dl_at_z(self, z):
        """
        Returns luminosity distance in Mpc given distance and cosmological parameters

        Parameters
        ----------
        z : np.array (or matrix)
            Cosmological redshift
        H0 : float
            Hubble constant today in km/Mpc/s
        """

        return _np.nan_to_num(10.**_splev(_np.log10(z),self.interp_z_to_dl,ext=0))

    def z_at_dl(self,dl):
        """
        Given a luminosity distance, this method returns
        the redshift

        Parameters
        ----------
        dl: _np.array
            Luminosity distance in Mpc
        """

        if not isinstance(dl,_np.ndarray):
            dl = _np.array([dl])

        if self.astropy_conv:
            z_ret = _np.array([_z_at_value(self.astropy_cosmology.luminosity_distance,
                d*_u.Mpc,zmax=self.zmax) for d in dl])
        else:
            z_ret = _np.nan_to_num(10.**_splev(_np.log10(dl),self.interp_dl_to_z,ext=0))
        return z_ret

    def dVc_by_dz(self,z):
        """
        Returns the differential in comoving volume in Units of :math:`{\\rm Gpc}^3`

        Parameters
        ----------
        z: _np.array
            Redshift

        """

        return _np.nan_to_num(10.**_splev(_np.log10(z),self.interp_dvc_dz,ext=0))

    def dL_by_dz(self, z,dl=None):
        """
        Calculates the d_dL/dz for this cosmology

        Parameters
        ----------
        z: _np. arrays
            Redshift
        """
        speed_of_light = _constants.c.to('km/s').value
        # Calculate the Jacobian of the luminosity distance w.r.t redshift
        if dl is None:
            return self.dl_at_z(z)/(1+z) + speed_of_light*(1+z)/(self.astropy_cosmology.H0.value*self.Efunc(z))
        else:
            return dl/(1+z) + speed_of_light*(1+z)/(self.astropy_cosmology.H0.value*self.Efunc(z))

    def Efunc(self,z):
        """
        Returns the :math:`E(z)=\\sqrt{\\Omega_{m,0}(1+z)^3+\\Omega_{\\Lambda}(1+z)^{3(1+w_0)}}` function for this cosmology

        Parameters
        ----------
        z: _np.array
            Redshift
        """

        return _np.sqrt(self.astropy_cosmology.Om0*_np.power(1+z,3)+(1-self.astropy_cosmology.Om0)*_np.power(1+z,3*(1+self.astropy_cosmology.w0)))


class w0waflatLCDM(object):
    """
    A Class to work with flat w0waLCDM cosmology
    """

    def __init__(self, Omega_m=0.308,H0=67,w0=-1.0,wa=0.,zmax=10, astropy_conv=False):
        """ Initialize the cosmology

        Parameters
        ----------
        Omega_m : float
            Fraction of matter energy
        H0 : float
            Hubble constant today in km/Mpc/s
        w0 : float
            Dark energy EOS parameter
        wa : float
            Evolving parameter EOS
        zmax : float
            Maximum redshift used for the cosmology
        astropy_conv : boolean
            Wether luminosity distance - redshift conversions are done with astropy
        """

        self.model_params = ['Om0','H0','w0','wa']
        self.astropy_cosmology = _Flatw0waCDM(Om0=Omega_m,H0=H0,w0=w0,wa=wa)
        self.zmax=zmax
        self.astropy_conv=astropy_conv

        # Calculate a lookup tables for the: luminosity distance, redshift and differential of comoving volume
        z_array = _np.logspace(-4,_np.log10(self.zmax),  2500)
        dl_trials = _np.log10(self.astropy_cosmology.luminosity_distance(z_array).value)
        dvc_dz_trials =  _np.log10(4*_np.pi*self.astropy_cosmology.differential_comoving_volume(z_array).to(_u.Gpc**3/_u.sr).value)

        z_array = _np.log10(z_array)

        # Interpolate the lookup tables
        self.interp_dvc_dz = _splrep(z_array,dvc_dz_trials)
        self.interp_dl_to_z = _splrep(dl_trials,z_array)
        self.interp_z_to_dl = _splrep(z_array,dl_trials)

    def dl_at_z(self, z):
        """
        Returns luminosity distance in Mpc given distance and cosmological parameters

        Parameters
        ----------
        z : np.array (or matrix)
            Cosmological redshift
        H0 : float
            Hubble constant today in km/Mpc/s
        """

        return _np.nan_to_num(10.**_splev(_np.log10(z),self.interp_z_to_dl,ext=0))

    def z_at_dl(self,dl):
        """
        Given a luminosity distance, this method returns
        the redshift

        Parameters
        ----------
        dl: _np.array
            Luminosity distance in Mpc
        """

        if not isinstance(dl,_np.ndarray):
            dl = _np.array([dl])

        if self.astropy_conv:
            z_ret = _np.array([_z_at_value(self.astropy_cosmology.luminosity_distance,
                d*_u.Mpc,zmax=self.zmax) for d in dl])
        else:
            z_ret = _np.nan_to_num(10.**_splev(_np.log10(dl),self.interp_dl_to_z,ext=0))
        return z_ret

    def dVc_by_dz(self,z):
        """
        Returns the differential in comoving volume in Units of :math:`{\\rm Gpc}^3`

        Parameters
        ----------
        z: _np.array
            Redshift

        """

        return _np.nan_to_num(10.**_splev(_np.log10(z),self.interp_dvc_dz,ext=0))

    def dL_by_dz(self, z,dl=None):
        """
        Calculates the d_dL/dz for this cosmology

        Parameters
        ----------
        z: _np. arrays
            Redshift
        """
        speed_of_light = _constants.c.to('km/s').value
        # Calculate the Jacobian of the luminosity distance w.r.t redshift
        if dl is None:
            return self.dl_at_z(z)/(1+z) + speed_of_light*(1+z)/(self.astropy_cosmology.H0.value*self.Efunc(z))
        else:
            return dl/(1+z) + speed_of_light*(1+z)/(self.astropy_cosmology.H0.value*self.Efunc(z))

    def Efunc(self,z):
        """
        Returns the :math:`E(z)=\\sqrt{\\Omega_{m,0}(1+z)^3+\\Omega_{\\Lambda}(1+z)^{3(1+w_0+w_a)} e^{-3 w_a z/(1+z)}}` function for this cosmology

        Parameters
        ----------
        z: _np.array
            Redshift
        """

        return _np.sqrt(self.astropy_cosmology.Om0*_np.power(1+z,3.)
        +(1-self.astropy_cosmology.Om0)*_np.power(1+z,3*(1+self.astropy_cosmology.w0+self.astropy_cosmology.wa))*_np.exp(-3*self.astropy_cosmology.wa*z/(1+z)))
