import numpy as _np
import seaborn as sns
import matplotlib.pyplot as _plt
import bilby as _bilby
import progressbar
from scipy.interpolate import interp1d as _interp1d
import copy as _copy

from ..cosmologies import flatLCDM as _flatLCDM
from ..cosmologies import w0flatLCDM as _w0flatLCDM
from ..cosmologies import w0waflatLCDM as _w0waflatLCDM

from ..priors.mass import mass_prior as _mass_prior
from ..priors.redshift import redshift_prior as _redshift_prior

__all__ = ['posterior_predictive_check','plot_CL']

def plot_CL(matrix,CL,kind='cdf',ax=None, fill_betweenx_kwargs={},matrix_2=None,color_ppplot=None):

    if ax is None:
        fig, ax = _plt.subplots(1,1,dpi=100)

    if kind == 'cdf':

        ax.fill_betweenx(y = _np.linspace(0,1,len(matrix[0,:])),
                x1 = _np.percentile(_np.sort(matrix,axis=1),50-0.5*CL,axis=0),
                x2 = _np.percentile(_np.sort(matrix,axis=1),50+0.5*CL,axis=0),
                **fill_betweenx_kwargs)

        ax.plot(_np.percentile(_np.sort(matrix,axis=1),50,axis=0),
        _np.linspace(0,1,len(matrix[0,:])),ls='--',c='k',alpha=1.)
        ax.plot(_np.percentile(_np.sort(matrix,axis=1),50-0.5*CL,axis=0),
        _np.linspace(0,1,len(matrix[0,:])),c='gray',alpha=1.)
        ax.plot(_np.percentile(_np.sort(matrix,axis=1),50+0.5*CL,axis=0),
        _np.linspace(0,1,len(matrix[0,:])),c='gray',alpha=1.)

        ax.set_ylabel('CDF')
        ax.legend()

    elif kind == 'ppplot':

        matrix_inj = matrix
        matrix_det = matrix_2

        matrix_inj = _np.sort(matrix_inj,axis=1)
        matrix_det = _np.sort(matrix_det,axis=1)

        sum_inj=_np.sum(matrix_inj,axis=1)
        sum_det=_np.sum(matrix_det,axis=1)

        min_val = _np.min(_np.vstack([matrix_inj,matrix_det]))
        max_val = _np.max(_np.vstack([matrix_inj,matrix_det]))

        arr_eval = _np.linspace(min_val,max_val,1000,endpoint=True)

        cdf_inj_matrix = _np.ones([matrix_inj.shape[0],len(arr_eval)])
        cdf_det_matrix = _np.ones([matrix_inj.shape[0],len(arr_eval)])

        for ip in range(matrix_inj.shape[0]):
            cdf_inj=_np.cumsum(matrix_inj[ip,:])/sum_inj[ip]
            cdf_det=_np.cumsum(matrix_det[ip,:])/sum_det[ip]

            interp_inj=_interp1d(matrix_inj[ip,:],cdf_inj,fill_value=(0,1),bounds_error=False)
            interp_det=_interp1d(matrix_det[ip,:],cdf_det,fill_value=(0,1),bounds_error=False)

            cdf_inj_matrix[ip,:] = interp_inj(arr_eval)
            cdf_det_matrix[ip,:] = interp_det(arr_eval)

        for ip in range(int(matrix_inj.shape[0]*CL/100.)):
            ax.plot(cdf_inj_matrix[ip,:],cdf_det_matrix[ip,:],
            c=color_ppplot,alpha=10./matrix_inj.shape[0])

        ax.plot(_np.percentile(cdf_inj_matrix,50,axis=0),
        _np.percentile(cdf_det_matrix,50,axis=0),ls='solid',c='k',alpha=1.,label='median')

        ax.plot(_np.linspace(0,1,10,endpoint=True),
        _np.linspace(0,1,10,endpoint=True),ls='-.',c='gray',alpha=1.,label='Reference')

        ax.legend()

    return ax

def posterior_predictive_check(pos_samples_dict,injections,result,mass_model, cosmo_model,
rates_model, Nsamp_mcmc=1000, subset = None, kind='cdf', N_sample_ev=5000,N_sample_inj=10000,fig_ax=None,CL=90.):

    samples_dict = result.posterior

    cpalette_inj = sns.color_palette("dark")
    cpalette_det = sns.color_palette("pastel")

    if subset is not None:
        analysis_events = subset
    else:
        analysis_events = list(pos_samples_dict.keys())

    list_param = list(samples_dict.keys())

    bar = progressbar.ProgressBar()
    m1_ev_tot = []
    m2_ev_tot = []
    z_ev_tot = []
    m1_in = []
    m2_in = []
    z_in = []

    if kind == 'ppplot':
        N_sample_inj = len(analysis_events)*N_sample_ev
        print('Overwriting samples of injections to '+str(N_sample_inj))

    loops = {event:0 for event in analysis_events}

    for i in bar(range(Nsamp_mcmc)):

        dic_pop_params = {ll:samples_dict[ll][i] for ll in list_param}
        mp_model = _mass_prior(name=mass_model,hyper_params_dict = dic_pop_params)
        

        if cosmo_model == 'flatLCDM':
            cosmo = _flatLCDM(Omega_m=samples_dict['Om0'][i],H0=samples_dict['H0'][i])
        elif cosmo_model == 'w0flatLCDM':
            cosmo = _w0flatLCDM(Omega_m=samples_dict['Om0'][i],H0=samples_dict['H0'][i],w0=samples_dict['w0'][i])
        elif cosmo_model == 'w0waflatLCDM':
            cosmo = _w0waflatLCDM(Omega_m=samples_dict['Om0'][i],H0=samples_dict['H0'][i],w0=samples_dict['w0'][i],wa=samples_dict['wa'][i])

        zp_model = _redshift_prior(cosmo,name=rates_model,dic_param=dic_pop_params)
        
        injections.update_VT(mp_model,zp_model)
        
        m1s,m2s,z,tot_weight,_ = injections.return_reweighted_injections(new_samples = N_sample_inj)
        m1_in.append(m1s)
        m2_in.append(m2s)
        z_in.append(z)

        m1_ev = []
        m2_ev = []
        z_ev = []

        for event in analysis_events:

            m1s,m2s,z,tot_weight,_ = pos_samples_dict[event].return_reweighted_samples(mp_model, zp_model,samples=N_sample_ev)
            m1_ev.append(m1s)
            m2_ev.append(m2s)
            z_ev.append(z)

        m1_ev_tot.append(_np.hstack(m1_ev))
        m2_ev_tot.append(_np.hstack(m2_ev))
        z_ev_tot.append(_np.hstack(z_ev))

    detected_matrix = {'mass_1_source':_np.vstack(m1_ev_tot),
    'mass_2_source':_np.vstack(m2_ev_tot),
    'redshift':_np.vstack(z_ev_tot) }

    injection_matrix  = {'mass_1_source':_np.vstack(m1_in),
    'mass_2_source':_np.vstack(m2_in),
    'redshift':_np.vstack(z_in)}

    if fig_ax is None:
        fig,ax = _plt.subplots(1,3,dpi=100,figsize=(15,5))
    else:
        fig = fig_ax[0]
        ax = fig_ax [1]

    for i,key in enumerate(list(injection_matrix.keys())):

        if kind == 'cdf':

            matrix = injection_matrix[key]
            ax[i]=plot_CL(matrix,CL,kind='cdf',ax=ax[i],
            fill_betweenx_kwargs={'color':cpalette_inj[i+1], 'alpha':0.5,'label':'Expected'})

            matrix = detected_matrix[key]
            ax[i]=plot_CL(matrix,CL,kind='cdf',ax=ax[i],
            fill_betweenx_kwargs={'color':cpalette_det[-i], 'alpha':0.5,'label':'Detected'})

            ax[i].set_ylabel('CDF')
            ax[i].set_xlabel(key)
            ax[i].legend()

        elif kind == 'ppplot':

            matrix_inj = injection_matrix[key]
            matrix_det = detected_matrix[key]

            ax[i]=plot_CL(matrix_inj,CL,kind='ppplot',ax=ax[i],matrix_2=matrix_det,color_ppplot=cpalette_inj[i+1])

            ax[i].set_ylabel('Detected distribution')
            ax[i].set_xlabel('Injection distribution')
            ax[i].set_title(key)
            ax[i].legend()

    _plt.tight_layout()

    return detected_matrix,injection_matrix,fig,ax
