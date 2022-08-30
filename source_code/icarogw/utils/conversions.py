"""
A Module with some utilites to convert between source frame and detector frame
"""

import numpy as _np

__all__=['source_to_detector_jacobian','detector_to_source_jacobian',
'source_frame_to_detector_frame','detector_frame_to_source_frame']

def source_to_detector_jacobian(dl,cosmo):
    """
    Calculates the soure frame to detector frame Jacobian d_sour/d_det

    Parameters
    ----------
    dl: _np.array
        Luminosity distance in Mpc
    cosmo:  class from the cosmology module
        Cosmology class from the cosmology module
    """
    z_samples = cosmo.z_at_dl(dl)
    jacobian = detector_to_source_jacobian(z_samples, cosmo,dl=dl)
    return 1./jacobian

def detector_to_source_jacobian(z, cosmo,dl=None):
    """
    Calculates the detector frame to source frame Jacobian d_det/d_sour

    Parameters
    ----------
    z: _np. arrays
        Redshift
    cosmo:  class from the cosmology module
        Cosmology class from the cosmology module
    """
    jacobian = _np.power(1+z,2)*cosmo.dL_by_dz(z,dl=dl)
    return jacobian

def source_frame_to_detector_frame(cosmo,ms_1,ms_2,redshift_samples):
    """
    Converts detector frame masses and luminosity distance to source frame masses and redshift.

    Parameters
    ----------
    cosmology: Cosmology class
        This is the cosmology class from the cosmology module
    ms_1: _np.array
        Array of first mass at the detector frame (solar masses)
    ms_2: _np.array
        Array of second mass at the detector frame (solar masses)
    redshift_samples: _np.arrray
        Array of luminosity distance samples at detector frame (in Mpc)

    Returns
    -------
    md1: _np.array
        Detector frame mass 1 in Msol
    md2: _np.array
        Detector frame mass 2 in Msol
    redshift_samples: _np.array
        cosmological redshift
    """

    distance_samples = cosmo.dl_at_z(redshift_samples)
    md1 = ms_1*(1+redshift_samples)
    md2 = ms_2*(1+redshift_samples)

    return md1, md2, distance_samples

def detector_frame_to_source_frame(cosmo,md_1,md_2,distance_samples):
    """
    Converts detector frame masses and luminosity distance to source frame masses and redshift.

    Parameters
    ----------
    cosmology: Cosmology class
        This is the cosmology class from the cosmology module
    md_1: _np.array
        Array of first mass at the detector frame (solar masses)
    md_2: _np.array
        Array of second mass at the detector frame (solar masses)
    distance_samples: _np.arrray
        Array of luminosity distance samples at detector frame (in Mpc)

    Returns
    -------
    ms_1: _np.array
        Array of first mass at the source frame (solar masses)
    ms_2: _np.array
        Array of second mass at the source frame (solar masses)
    redshift_samples: _np.arrray
        Array of redshift (in Mpc)
    """

    z_samples = cosmo.z_at_dl(distance_samples)
    ms1 = md_1/(1+z_samples)
    ms2 = md_2/(1+z_samples)

    return ms1, ms2, z_samples
