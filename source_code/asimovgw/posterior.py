import numpy as _np
import corner as _corner
from scipy.stats import gaussian_kde as _gaussian_kde
import matplotlib.pyplot as _plt
from scipy.special import logsumexp as _logsumexp


class list_of_posteriors(object):
    """
    This is a wrapper class to collect all posterior samples in one place. I might think to add methods to clean space

    Parameters
    ----------
    posterior_list: list
        List of posterior objects from this module
    """
    def __init__(self,posterior_list):
        self.posterior_list={posterior_list[i].name:posterior_list[i] for i in range(len(posterior_list))}
        self.posterior_names=list(self.posterior_list.keys())
        self.binary_vars=self.posterior_list[self.posterior_names[0]].binary_vars

class posterior(object):
    """
    Posterior class to handle GW events

    Parameters
    ----------
    name: str
        Name to assign to the binary
    binary_params: dict
        Dictionary containing the posterior samples. Each field must be named after the varialbe and should contain a dictionary with
        'values' field containing the array, 'tex_label' containing the tex string to plot axis, 'tex_var' the tex variable without unit of measures,
        example

        ```
        popbin={'mass_1_source':{'values':pos['mass_1_source'],'tex_label':r'$m_1 {\rm [M_{\odot}]}$','tex_var':r'm_1'},
        'mass_2_source': {'values':pos['mass_2_source'],'tex_label':r'$m_2 {\rm [M_{\odot}}]$','tex_var':r'm_2'},
        'redshift': {'values':pos['redshift'],'tex_label':r'$z$','tex_var':r'z'}}
        ```
    prior_to_rem: array
        Prior to remove when you are summing over posterior samples.

    """

    def __init__(self,name,binary_params,prior_to_rem):
        self.binary_params=binary_params
        self.binary_vars=list(self.binary_params.keys())
        self.name=name
        self.prior_to_rem=prior_to_rem

    def find_pop_extremes(self):
        """
        This method finds and returns the minimum and maximum value for each parameter in the posterior samples.
        """
        vars=self.binary_vars
        minmax={}
        for var in vars:
            minmax[var]={'min':_np.inf,'max':-_np.inf}
            minmax[var]['min']=self.binary_params[var]['values'].min()
            minmax[var]['max']=self.binary_params[var]['values'].max()

        return minmax

    def kernelize(self,kernel_mode='kde',bins=None,**kwargs):
        """
        This method express and initialize the posterior as an ndhistogram density or KDE fitted density

        Parameters
        ----------
        kernel_mode: str (optional)
            How to fit the posterior samples, either histogram or kde.
        **kwargs: Any argument to pass to the kernel fitting (gaussian_kde or ndhistogram)
        """
        self.kernel_mode=kernel_mode
        if kernel_mode=='kde':
            self.minmax=self.find_pop_extremes()
            self.kernel=_gaussian_kde(_np.vstack([self.binary_params[var]['values'] for var in self.binary_vars]),**kwargs)
        elif kernel_mode=='histogram':
            toplot = _np.column_stack([self.binary_params[var]['values']for var in self.binary_vars])
            bb=[bins[var]  for var in self.binary_vars]
            self.kernel,self.common_edges=_np.histogramdd(toplot,density=True,bins=bb,**kwargs)
            return
        else:
            raise ValueError

    def histogram_samples(self,**kwargs):
        """
        This method histogram with corner the parameters

        Parameters
        ----------
        **kwargs: Any kwargs you would pass to corner.corner

        Returns
        -------
        Fig handling
        """
        toplot = _np.column_stack([self.binary_params[var]['values'] for var in self.binary_vars])
        labels = [self.binary_params[var]['tex_label'] for var in  self.binary_vars]
        return _corner.corner(toplot,labels=labels,**kwargs)

    def calculate_model_match(self,syn_model):
        """
        This method calculates the log of the match, i.e. montecarlo integral given a population model
        of the GW likelihood. This particular integral is done over astrophysical rates.

        Parameters
        ----------
        syn_model: obj
            Synthetic_population object from its module

        Returns
        -------
        log of the likelihood integral
        """
        prior_to_rem=syn_model.prior_to_rem
        ## Add a check so that the syn_model and binary parameter_arr are the same
        if self.binary_vars.sort()!=syn_model.binary_vars.sort():
            raise ValueError('Syntehtic population and posterior must have the same parameters')

        if self.kernel_mode=='kde':

            vars=self.binary_vars
            list_of_index=[]
            total_samples_j=len(syn_model.binary_params[vars[0]]['values'])
            idxfinal=_np.where(syn_model.binary_params[vars[0]]['values']!=_np.nan)[0]

            for k in range(len(vars)):
                name_var=vars[k]
                parameter_arr=syn_model.binary_params[name_var]['values']
                idx0=_np.where((parameter_arr>=self.minmax[name_var]['min'])
                                 & (parameter_arr<=self.minmax[name_var]['max']))[0]
                idxfinal=_np.intersect1d(idxfinal,idx0)
                if len(idxfinal)==0:
                    return -_np.inf

            positions=_np.vstack([syn_model.binary_params[var]['values'][idxfinal] for var in self.binary_vars])
            logpdfs=self.kernel.logpdf(positions)-_np.log(prior_to_rem[idxfinal])
            log_lij=_logsumexp(logpdfs)-_np.log(len(syn_model.binary_params[vars[0]]['values']))

        elif self.kernel_mode=='histogram':
            vars=self.binary_vars
            list_of_index=[]
            total_samples_j=len(syn_model.binary_params[vars[0]]['values'])
            idxfinal=_np.where(syn_model.binary_params[vars[0]]['values']!=_np.nan)[0]

            for k in range(len(self.common_edges)):
                name_var=vars[k]
                parameter_arr=syn_model.binary_params[name_var]['values']
                idx0=_np.where((parameter_arr>=self.common_edges[k][0])
                 & (parameter_arr<=self.common_edges[k][-1]))[0]
                idxfinal=_np.intersect1d(idxfinal,idx0)
                if len(idxfinal)==0:
                    return -_np.inf

            for k in range(len(self.common_edges)):
                name_var=vars[k]
                parameter_arr=syn_model.binary_params[name_var]['values'][idxfinal]
                toc=_np.reshape(_np.digitize(parameter_arr,self.common_edges[k])-1,-1)
                toc[toc==(len(self.common_edges[k])-1)]=len(self.common_edges[k])-2
                list_of_index.append(toc)
            list_of_index=tuple(list_of_index)

            log_lij=_logsumexp(_np.log(self.kernel[list_of_index])-_np.log(prior_to_rem[idxfinal]))-_np.log(total_samples_j)

        return log_lij



    def histogram_kernel(self,**kwargs):
        """
        This method histogram the fitted kernel

        Parameters
        ----------
        **kwargs: Any kwargs you would pass to corner.corner

        Returns
        -------
        Fig handling
        """

        if self.kernel_mode=='kde':
            samples=self.kernel.resample(10000).T
            labels = [self.binary_params[var]['tex_label'] for var in  self.binary_vars]
            return _corner.corner(samples,labels=labels,**kwargs)

        elif self.kernel_mode=='histogram':
            edges=self.common_edges
            vars=self.binary_vars
            counts=self.kernel
            dimensions=len(edges)
            bincenters=[(edges[i][0:-1:]+edges[i][1::])*0.5 for i in range(dimensions)]
            bintops=[edges[i][1::]*0.5 for i in range(dimensions)]
            bitbottoms=[edges[i][0:-1:]*0.5 for i in range(dimensions)]

            meshgrid_center=_np.meshgrid(*bincenters,indexing='ij')
            meshgrid_top=_np.meshgrid(*bintops,indexing='ij')
            meshgrid_bottoms=_np.meshgrid(*bitbottoms,indexing='ij')
            dvolumes=[meshgrid_top[i]-meshgrid_bottoms[i] for i in range(dimensions)]
            DV=_np.ones_like(meshgrid_top[0])
            for i in range(dimensions):
                DV*=dvolumes[i]

            fig,ax=_plt.subplots(dimensions,dimensions,figsize=(7.0,6.5))
            _plt.subplots_adjust(wspace=0., hspace=0.0)

            for index, x in _np.ndenumerate(ax):
                ax[index].tick_params(axis='both', which='both', labelsize=3.5,length=1.0,width=0.25)
                if index[0]==index[1]:
                    xplot=(edges[index[0]][0:-1:]+edges[index[0]][1::])*0.5
                    summize=[]
                    for jax in range(dimensions):
                        if  (jax!=index[0]):
                            summize.append(jax)
                    if len(summize)==1:
                        summize=summize[0]
                    else:
                        summize=tuple(summize)
                    marginal=_np.sum(DV*counts,axis=summize)
                    ax[index].plot(xplot,marginal)
                    ax[index].set_ylim([0,_np.max(marginal)*1.2])
                    ax[index].set_yticklabels([])
                    if index[0]==(dimensions-1):
                        ax[index].set_xlabel(self.binary_params[vars[index[0]]]['tex_label'],fontsize=3.5)
                    else:
                        ax[index].set_xticklabels([])


                elif index[1]>index[0]:
                    fig.delaxes(ax[index])

                else:
                    Xplot,Yplot=_np.meshgrid(edges[index[0]],edges[index[1]])
                    summize=[]
                    for jax in range(dimensions):
                        if  (jax!=index[0]) & (jax!=index[1]):
                            summize.append(jax)
                    if len(summize)==0:
                        marginal=DV*counts
                    else:
                        marginal=_np.sum(DV*counts,axis=tuple(summize))

                    pcm=ax[index].pcolormesh(Yplot,Xplot,marginal,vmin=0,vmax=1.2*_np.max(marginal))

                    if index[1]==0:
                        ax[index].set_ylabel(self.binary_params[vars[index[0]]]['tex_label'],fontsize=3.5)
                    else:
                        ax[index].set_yticklabels([])

                    if index[0]==(dimensions-1):
                        ax[index].set_xlabel(self.binary_params[vars[index[1]]]['tex_label'],fontsize=3.5)
                    else:
                        ax[index].set_xticklabels([])


                    #cbar=fig.colorbar(pcm,ax=ax[index])
                    #cbar.ax.tick_params(labelsize=3.5,length=1.0,width=0.25)
            _plt.tight_layout()
            return fig
