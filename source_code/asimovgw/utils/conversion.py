import numpy as _np
import astropy.units as _u
from astropy.cosmology import z_at_value as _z_at_value



def ndangles2cartesian(ndangles):
    """
    This function converts an array of angles to cartesian coordinates.

    Parameters
    ----------
    ndangles: np.ndarray
        Array of N angles in radians. The angles are defined between a reference axis referred as 1
        The angles [phi_21,phi_31,...,phi_N_1] should go from [0,pi] the N angle from [0,2pi]

    Returns
    -------
    Cartesian components [x2,x3,....,xN,x1]
    """
    x=_np.ones(len(ndangles)+1)
    for i in range(len(ndangles)):
        x[i]*=_np.cos(ndangles[i])
        x[i+1]*=_np.prod(_np.sin(ndangles[:i+1:]))
    return x

def masses2massratio(m1,m2):
    """
    Returns mass ratio

    Parameters
    ----------
    m1: np.ndarray
        mass 1 in solar masses
    m2: np.ndarray
        mass 2 in solar masses

    Returns
    -------
    Mass ratio m2/m1
    """
    return m2/m1

def calculate_chis(a1,a2,m1,m2,cost1,cost2):
    """
    Returns mass chi eff and chip given spins, masses and cosine of tilt angles

    Parameters
    ----------
    a1,a2: np.ndarray
        spin magnitude parameters of the first and secondary object in [0,1]
    m1,m2: np.ndarray
        masses in solar masses of the primary and secondary object in solar masses
    cost1,cost2: np.array
        Cosine of the title angle between the spin and orbital angular momentum for the two bodies.

    Returns
    -------
    chieff and chip
    """
    q = masses2massratio(m1,m2)
    chieff=(a1*cost1+q*a2*cost2)/(1+q)
    chip1=a1*_np.sin(_np.arccos(cost1))
    chip2=((3+4.*q)/(4+3.*q))*q*a2*_np.sin(_np.arccos(cost2))
    return chieff, _np.max(_np.stack([chip1,chip2]),axis=0)

def sourceframe2detectorframe(m1s,m2s,z,cosmology):
    """
    Converts source frame masses and redshift to detector frame masses given a cosmology

    Parameters
    ----------
    m1s, m2s, z: np.ndarray
        mass 1, mass 2 in solar masses and redshift of the GW source
    cosmology: astropy cosmology
        Astropy cosmology to use for conversion

    Returns
    -------
    Detector frame m1, m2 (in solar masses) and luminosity distance in Mpc
    """
    return m1s*(1+z),m2s*(1+z),cosmology.luminosity_distance.to(_u.Mpc).value

def detectorframe2sourceframe(m1d,m2d,dl,cosmology,zmax=100):
    """
    Converts detector frame masses and luminosity distance to source frame masses given a cosmology

    Parameters
    ----------
    m1d, m2d, dl: np.ndarray
        mass 1, mass 2 in solar masses and luminosity distance in Mpc of the GW source
    cosmology: astropy cosmology
        Astropy cosmology to use for conversion
    zmax: float (optional)
        Maximum redshift at which to look for the luminosity distance value

    Returns
    -------
    Source frame m1, m2 (in solar masses) and redshift
    """
    z=dl2z(dl,cosmology,zmax=zmax)
    return m1d/(1+z),m2d/(1+z),z

def dl2z(dl,cosmology,zmax=100):
    """
    Converts luminosity distance to redshift

    Parameters
    ----------
    dl: np.ndarray
        luminosity distance in Mpc of the GW source
    cosmology: astropy cosmology
        Astropy cosmology to use for conversion
    zmax: float (optional)
        Maximum redshift at which to look for the luminosity distance value

    Returns
    -------
    redshift array.
    """
    return _np.array([_z_at_value(cosmology.luminosity_distance,dd*_u.Mpc,zmax=zmax)  for dd in dl])
