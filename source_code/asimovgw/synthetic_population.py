import numpy as _np
import corner as _corner
import matplotlib.pyplot as _plt
from scipy.special import logsumexp as _logsumexp
from scipy.stats import poisson as _poisson
from scipy.special import logsumexp as _logsumexp
import pickle as _pickle
from scipy.stats import gaussian_kde as _gaussian_kde


class list_of_populations(object):
    """
    Wrapper for a list of synthetic models. It contains several methods used when doing calculus with a set of models.

    Parameters
    ----------
    population_list: list
        List of populations objects from this module
    """

    def __init__(self,population_list):
        self.population_list={population_list[i].name:population_list[i] for i in range(len(population_list))}
        self.population_names=list(self.population_list.keys())
        plist=[]
        # Renormalize the prior
        for key in self.population_list.keys():
            plist.append(self.population_list[key].prior_model)
        plist=_np.array(plist)
        plist/=plist.sum()
        for i,key in enumerate(self.population_list.keys()):
            self.population_list[key].prior_model=plist[i]

        self.binary_vars=self.population_list[self.population_names[0]].binary_vars

    def combine_list_of_pop(self,lambdas,Ntogen=80000):
        Ntot_true=[]
        Nsim_true=[]
        for i,name in enumerate(self.population_names):
            Ntot_true.append(self.population_list[name].Ntotal*lambdas[name])
            Nsim_true.append(self.population_list[name].Nsim)
        Ntot_true,Nsim_true=_np.array(Ntot_true),_np.array(Nsim_true)
        relprobpoint=(Ntot_true/Nsim_true)*(Nsim_true.sum()/Ntot_true.sum())

        pop_dicto={}
        support_dicto={}
        for var in self.binary_vars:
            pop_dicto[var]={'values':[],'tex_label':self.population_list[self.population_names[0]].binary_params[var]['tex_label']
            ,'tex_var':self.population_list[self.population_names[0]].binary_params[var]['tex_var']}
            support_dicto[var]=[]
        support_dicto['detected']=[]
        support_dicto['probs']=[]

        for i,name in enumerate(self.population_names):
            for var in self.binary_vars:
                support_dicto[var].append(self.population_list[name].binary_params[var]['values'])
            support_dicto['detected'].append(self.population_list[name].ilabel_det)
            support_dicto['probs'].append(_np.ones_like(self.population_list[name].ilabel_det)*relprobpoint[i])

        for var in self.binary_vars:
            support_dicto[var]=_np.hstack(support_dicto[var])
        support_dicto['detected']=_np.hstack(support_dicto['detected'])
        support_dicto['probs']=_np.hstack(support_dicto['probs'])
        support_dicto['probs']/=support_dicto['probs'].sum()

        chx=_np.random.choice(len(support_dicto['probs']),p=support_dicto['probs'],replace=True,size=Ntogen)

        for var in self.binary_vars:
            pop_dicto[var]['values']=support_dicto[var][chx]

        pop_true=synthetic_population('Combined',Ntot_true.sum()
        ,pop_dicto,prior_to_rem=_np.ones(len(pop_dicto[self.binary_vars[0]]['values'])),prior_model=None,Tobs=None)
        pop_true.apply_selection_cut(ilabel_det=support_dicto['detected'][chx])
        return pop_true




    def find_pop_extremes(self):
        """
        This method finds and returns the minimum and maximum value for each parameter in the population samples.
        """
        vars=self.binary_vars
        minmax={}
        for var in vars:
            minmax[var]={'min':_np.inf,'max':-_np.inf}
            for key in self.population_list.keys():
                minmax[var]['min']=_np.min([minmax[var]['min'],self.population_list[key].binary_params[var]['values'].min()])
                minmax[var]['max']=_np.max([minmax[var]['max'],self.population_list[key].binary_params[var]['values'].max()])

        return minmax

    def calculate_log_pmodel_given_theta(self,bins,Nexp=None):
        """
        This method calculates and initialize under each population model the log of the probability of the model
        given a set of physical parameters p(phij|theta). It also initialize the null probability given (theta), e.g. none of the models studied.

        Parameters
        ----------
        bins: tuple
            Number of bins in which to divide the parameter space (#FIX check that for 1d model still works)
        Nexp: float (optional)
            Expected number of detections for the phenomenological model
        """
        vars=self.binary_vars
        self.pmodel_vars=vars
        minmax=self.find_pop_extremes()
        self.minmax=minmax

        edges={}
        if isinstance(bins[vars[0]],_np.ndarray):
            for i,var in enumerate(vars):
                edges[var]=bins[var]
        else:
            for i,var in enumerate(vars):
                edges[var]=_np.linspace(minmax[var]['min'],minmax[var]['max'],bins[var])

        for key in self.population_list.keys():
            self.population_list[key].p_model_given_theta,self.common_edges=self.population_list[key]._histogramdn(bins_dict=edges)

        norm_factor=[]
        for key in self.population_list.keys():

            if Nexp is not None:
                Nexp_phi=self.population_list[key].expected_detections
                logNpart=-Nexp_phi+Nexp*_np.log(Nexp_phi)-Nexp*_np.log(Nexp)+Nexp
            else:
                logNpart=0.

            norm_factor.append(_np.log(self.population_list[key].p_model_given_theta)+_np.log(self.population_list[key].prior_model)+logNpart)

            self.population_list[key].log_p_model_given_theta=_np.log(self.population_list[key].p_model_given_theta)+_np.log(self.population_list[key].prior_model*_np.exp(logNpart))

        norm_factor=_np.stack(norm_factor,axis=-1)
        log_norm_factor=_logsumexp(norm_factor,axis=-1)
        log_norm_factor=log_norm_factor


        self.p_null_model_given_theta=1.
        for key in self.population_list.keys():
            self.population_list[key].log_p_model_given_theta-=log_norm_factor
            self.population_list[key].log_p_model_given_theta[log_norm_factor==-_np.inf]=-_np.inf
            self.population_list[key].p_model_given_theta=_np.exp(self.population_list[key].log_p_model_given_theta)
            self.p_null_model_given_theta-=self.population_list[key].p_model_given_theta

        self.p_null_model_given_theta[self.p_null_model_given_theta<0]=0.
        self.log_p_null_model_given_theta=_np.log(self.p_null_model_given_theta)

    def calculate_phehom_match(self,list_of_phenom_samples,bins,list_of_Nexp=None):
        """
        This method calculates p(phi|x) marginalizing over all possible phenomenological models

        Parameters
        ----------
        list_of_phenom_samples: dict
            Dictionary containing the list of phenomenological samples for the parameters. Note that the field name of this dictiotnary
            should correspond with the names of the variables in the posterior samples
        bins: tuple
            Number of bins in which to divide the parameter space (#FIX check that for 1d model still works)
        list_of_Nexp: list of float (optional)
            List of expected number of detections for the phenomenological model. If none, we will consider a scale free prior.
        """
        vars=list(list_of_phenom_samples[0].keys())
        Npop=len(list_of_phenom_samples)

        out_dict={}

        for key in self.population_list.keys():
            out_dict[self.population_list[key].name]=0

        for i in range(len(list_of_phenom_samples)):
            if list_of_Nexp is not None:
                self.calculate_log_pmodel_given_theta(bins=bins,Nexp=list_of_Nexp[i])
            else:
                self.calculate_log_pmodel_given_theta(bins=bins)


            list_of_index=[]
            total_samples_j=len(list_of_phenom_samples[i][vars[0]])
            idxfinal=_np.where(list_of_phenom_samples[i][vars[0]]!=_np.nan)[0]

            for k in range(len(self.common_edges)):
                name_var=vars[k]
                parameter_arr=list_of_phenom_samples[i][name_var]
                idx0=_np.where((list_of_phenom_samples[i][vars[k]]>=self.minmax[vars[k]]['min'])
                 & (list_of_phenom_samples[i][vars[k]]<=self.minmax[vars[k]]['max']))[0]
                idxfinal=_np.intersect1d(idxfinal,idx0)

            if len(idx0)!=0:
                for k in range(len(self.common_edges)):
                    name_var=vars[k]
                    parameter_arr=list_of_phenom_samples[i][vars[k]][idxfinal]
                    list_of_index.append(_np.digitize(parameter_arr,self.common_edges[k])-1)
                list_of_index=tuple(list_of_index)
                for key in self.population_list.keys():
                    out_dict[self.population_list[key].name]+=_np.exp((_logsumexp(self.population_list[key].log_p_model_given_theta[list_of_index])-_np.log(total_samples_j)))/Npop

        out_dict['null']=1.-_np.sum([out_dict[var] for var in out_dict.keys()])

        return out_dict


    def plot_histograms_detected(self,name,**kwargs):
        """
        This method histogram the detected distribution of GW event

        Parameters
        ----------
        name: str
            name of the model to plot
        **kwargs: Any kwargs you would pass to corner.corner

        Returns
        -------
        Fig handling
        """
        plotted=False
        for key in self.population_list.keys():
            if self.population_list[key].name ==name:
                toplot = _np.column_stack([self.population_list[key].binary_params[var]['values'][self.population_list[key].detlabels] for var in self.pmodel_vars])
                labels = [self.population_list[key].binary_params[var]['tex_label'] for var in self.pmodel_vars]
                ran=[(self.minmax[var]['min'],self.minmax[var]['max']) for var in self.pmodel_vars]
                bin=[len(self.common_edges[jax]) for jax in range(len(self.common_edges))]
                return _corner.corner(toplot,bins=bin,range=ran,labels=labels,**kwargs)
        if not plotted: raise ValueError('No such model, check the name')

    def plot_histograms_astrophysical(self,name,**kwargs):
        """
        This method histogram the astrophysical distribution of GW event

        Parameters
        ----------
        name: str
            name of the model to plot
        **kwargs: Any kwargs you would pass to corner.corner

        Returns
        -------
        Fig handling
        """
        plotted=False
        for key in self.population_list.keys():
            if self.population_list[key].name ==name:
                toplot = _np.column_stack([self.population_list[key].binary_params[var]['values'] for var in self.pmodel_vars])
                labels = [self.population_list[key].binary_params[var]['tex_label'] for var in self.pmodel_vars]
                ran=[(self.minmax[var]['min'],self.minmax[var]['max']) for var in self.pmodel_vars]
                bin=[len(self.common_edges[jax]) for jax in range(len(self.common_edges))]
                return _corner.corner(toplot,bins=bin,range=ran,labels=labels,**kwargs)
        if not plotted: raise ValueError('No such model, check the name')


    def plot_pmodel_given_theta(self,name,bins,Nexp=None):
        """
        This method plot corner plots for p(phij|theta)

        Parameters
        ----------
        name: str
            name of the model to plot
        bins: tuple
            Number of bins in which to divide the parameter space (#FIX check that for 1d model still works)
        Nexp:  float (optional)
            Expected number of detections for the phenomenological model. If none, we will consider a scale free prior.

        Returns
        -------
        Fig handling
        """
        vars=self.binary_vars
        plotted=False
        self.calculate_log_pmodel_given_theta(bins=bins,Nexp=Nexp)

        for key in self.population_list.keys():
            if (self.population_list[key].name == name) | (name=='null'):
                if name!='null':
                    counts=self.population_list[key].p_model_given_theta
                else:
                    counts=self.p_null_model_given_theta
                edges=self.common_edges
                dimensions=len(edges)
                bincenters=[(edges[i][0:-1:]+edges[i][1::])*0.5 for i in range(dimensions)]
                fig,ax=_plt.subplots(dimensions,dimensions)
                _plt.subplots_adjust(wspace=0.1, hspace=0.1)

                for index, x in _np.ndenumerate(ax):
                    ax[index].tick_params(axis='both', which='both', labelsize=3.5,length=1.0,width=0.25)
                    if index[0]==index[1]:
                        xplot=(edges[index[0]][0:-1:]+edges[index[0]][1::])*0.5
                        summize=[]
                        for jax in range(dimensions):
                            if  (jax!=index[0]):
                                summize.append(jax)
                        if len(summize)==1:
                            summize=tuple(summize)
                        else:
                            summize=tuple(summize)
                        totnorm=_np.prod([self.p_null_model_given_theta.shape[jax] for jax in summize])
                        marginal=_np.sum(counts,axis=summize)

                        marginal/=totnorm
                        marginal[totnorm==0]=0.
                        ax[index].plot(xplot,marginal)
                        ax[index].set_ylim([0,1.2*_np.max(marginal)])
                        if index[0]==(dimensions-1):
                            ax[index].set_xlabel(self.population_list[key].binary_params[vars[index[0]]]['tex_label'],fontsize=3.5)
                        ax[index].set_ylabel(r'$p(\varphi_j|{:s})$'.format(self.population_list[key].binary_params[vars[index[0]]]['tex_var']),fontsize=3.5)

                    elif index[1]>index[0]:
                        fig.delaxes(ax[index])

                    else:
                        Xplot,Yplot=_np.meshgrid(edges[index[0]],edges[index[1]])
                        summize=[]
                        for jax in range(dimensions):
                            if  (jax!=index[0]) & (jax!=index[1]):
                                summize.append(jax)
                        if len(summize)==0:
                            totnorm=1
                            marginal=counts/totnorm
                        else:
                            summize=tuple(summize)
                            totnorm=_np.prod([self.p_null_model_given_theta.shape[jax] for jax in summize])

                            marginal=_np.sum(counts,axis=summize)
                            marginal/=totnorm
                        pcm=ax[index].pcolormesh(Yplot,Xplot,marginal,vmin=0,vmax=1)

                        cbar=fig.colorbar(pcm,ax=ax[index])
                        cbar.ax.tick_params(labelsize=3.5,length=1.0,width=0.25)
                        if index[1]==0:
                            ax[index].set_ylabel(self.population_list[key].binary_params[vars[index[0]]]['tex_label'],fontsize=3.5)
                        else:
                            ax[index].set_yticklabels([])

                        if index[0]==(dimensions-1):
                            ax[index].set_xlabel(self.population_list[key].binary_params[vars[index[1]]]['tex_label'],fontsize=3.5)
                        else:
                            ax[index].set_xticklabels([])
                plotted=True
                _plt.tight_layout()
                break
        if not plotted: raise ValueError('No such model, check the name')

class synthetic_population(object):
    """
    This is the basic class where to store simulations of syntethic binaries.

    Parameters
    ----------
    name: str
        Name of the model
    Ntotal: float
        Total number of binaries merging in Tobs in your model (up to the maximum redshift you simulated)
    binary_params: dict
        Dictionary containing the posterior samples. Each field must be named after the varialbe and should contain a dictionary with
        'values' field containing the array, 'tex_label' containing the tex string to plot axis, 'tex_var' the tex variable without unit of measures,
        example

        ```
        popbin={'mass_1_source':{'values':pos['mass_1_source'],'tex_label':r'$m_1 {\rm [M_{\odot}]}$','tex_var':r'm_1'},
        'mass_2_source': {'values':pos['mass_2_source'],'tex_label':r'$m_2 {\rm [M_{\odot}}]$','tex_var':r'm_2'},
        'redshift': {'values':pos['redshift'],'tex_label':r'$z$','tex_var':r'z'}}
        ```
    prior_to_rem: np.array
        1d array with same lenght as the binary provided in your simulatio with the prior to remove
        from the posterior samples of the GW. Needed for mixture analysis.
    Tobs: float (optional)
        Observing time, just to bookeping
    """

    def __init__(self,name,Ntotal,binary_params,prior_to_rem,prior_model=1.,Tobs=None):

        self.prior_to_rem=prior_to_rem
        self.name=name
        self.Ntotal=Ntotal
        self.Tobs=Tobs
        self.binary_vars=list(binary_params.keys())
        self.Nsim=len(binary_params[self.binary_vars[0]]['values'])
        self.prior_model=prior_model
        self.binary_params={var:binary_params[var] for var in binary_params.keys()}
        self.selection_effect=1.
        self.expected_detections=Ntotal
        self.ilabel_det=_np.where(self.binary_params[self.binary_vars[0]]['values']!=_np.nan)[0]
        self.detlabels=_np.where(self.ilabel_det)[0]

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
            toplot = _np.column_stack([self.binary_params[var]['values'] for var in self.binary_vars])
            bb=[bins[var]  for var in self.binary_vars]
            self.kernel,self.common_edges=_np.histogramdd(toplot,density=True,bins=bb,**kwargs)
            return
        else:
            raise ValueError

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

    def calculate_model_match(self,pos):
        """
        This method calculates the log of the match, i.e. montecarlo integral given a population model
        of the GW likelihood. This particular method performs the integral summing over posterior samples

        Parameters
        ----------
        pos: obj
            Posterior object from its module

        Returns
        -------
        log of the likelihood integral
        """
        prior_to_rem=pos.prior_to_rem
        ## Add a check so that the pos and binary parameter_arr are the same
        if self.binary_vars.sort()!=pos.binary_vars.sort():
            raise ValueError('Syntehtic population and posterior must have the same parameters')

        if self.kernel_mode=='kde':

            vars=self.binary_vars
            list_of_index=[]
            total_samples_j=len(pos.binary_params[vars[0]]['values'])
            idxfinal=_np.where(pos.binary_params[vars[0]]['values']!=_np.nan)[0]

            for k in range(len(vars)):
                name_var=vars[k]
                parameter_arr=pos.binary_params[name_var]['values']
                idx0=_np.where((parameter_arr>=self.minmax[name_var]['min'])
                                 & (parameter_arr<=self.minmax[name_var]['max']))[0]
                idxfinal=_np.intersect1d(idxfinal,idx0)
                if len(idxfinal)==0:
                    return -_np.inf

            positions=_np.vstack([pos.binary_params[var]['values'][idxfinal] for var in self.binary_vars])
            logpdfs=self.kernel.logpdf(positions)-_np.log(prior_to_rem[idxfinal])
            log_lij=_logsumexp(logpdfs)-_np.log(total_samples_j)

        elif self.kernel_mode=='histogram':
            vars=self.binary_vars
            list_of_index=[]
            total_samples_j=len(pos.binary_params[vars[0]]['values'])
            idxfinal=_np.where(pos.binary_params[vars[0]]['values']!=_np.nan)[0]

            for k in range(len(self.common_edges)):
                name_var=vars[k]
                parameter_arr=pos.binary_params[name_var]['values']
                idx0=_np.where((parameter_arr>=self.common_edges[k][0])
                 & (parameter_arr<=self.common_edges[k][-1]))[0]
                idxfinal=_np.intersect1d(idxfinal,idx0)
                if len(idxfinal)==0:
                    return -_np.inf

            for k in range(len(self.common_edges)):
                name_var=vars[k]
                parameter_arr=pos.binary_params[name_var]['values'][idxfinal]
                toc=_np.reshape(_np.digitize(parameter_arr,self.common_edges[k])-1,-1)
                toc[toc==(len(self.common_edges[k])-1)]=len(self.common_edges[k])-2
                list_of_index.append(toc)
            list_of_index=tuple(list_of_index)

            log_lij=_logsumexp(_np.log(self.kernel[list_of_index])-_np.log(prior_to_rem[idxfinal]))-_np.log(total_samples_j)

        return log_lij



    def generate_delta_like_injections(self,outdir,Ngen,sigma,Nsamp=5000,mode='uniform'):
        """
        This method generate and dump in pickles dictionaries ready to be provided in the posterior sample module,
         delta like posteior samples drawing from the population. Note that delta-like posteriors
        are approximated with either narrow uniform or gaussian distribution around the true value. This is not completely correct to evaluate selection biases
        but for narrow posteriors should not give problems.

        Parameters
        ----------
        outdier: str
            Path finishing with / where to save the injections
        Ngen: str
            Number of injections to generate
        sigma: dict
            Discitionary containing the posterior width (fractional) to use for each parameter
        Nsamp: int (optional)
             Number of samples to generate
        mode: str (optional)
            If to generate the posterior `uniform` or `gaussian`
        """

        for i in range(Ngen):
            chx=_np.random.choice(len(self.detlabels),size=1,replace=True)[0]
            inj_dict={}
            for var in self.binary_vars:
                val=self.binary_params[var]['values'][self.detlabels[chx]]
                if mode == 'uniform':
                    if val==0:
                        num=_np.random.uniform(low=-sigma[var],high=sigma[var],size=Nsamp)
                    else:
                        num=_np.random.uniform(low=val*(1.-sigma[var]),high=val*(1.+sigma[var]),size=Nsamp)
                elif mode == 'gaussian':
                    if val==0:
                        num=_np.random.randn(Nsamp)*sigma[var]
                    else:
                        num=_np.random.randn(Nsamp)*val*sigma[var]+val

                num=_np.reshape(num,-1)

                inj_dict[var]={'values':num,'tex_label':self.binary_params[var]['tex_label'],'tex_var':self.binary_params[var]['tex_var']}

            _pickle.dump(inj_dict,open(outdir+'injection_{:d}.p'.format(i),'wb'))


    def apply_selection_cut(self,ilabel_det):
        """
        Apply a selection cut and redefines selection effects, expected number of detections and so on

        Parameters
        ----------
        ilabel_det: nd.array
            Label array corresponding to the detected events
        """

        self.ilabel_det = ilabel_det
        self.detlabels=_np.where(ilabel_det)[0]
        self.selection_effect=len(self.detlabels)/self.Nsim
        self.expected_detections=self.selection_effect*self.Ntotal

    def histogram_detected_population(self,**kwargs):
        """
        This method histogram the detected distribution of GW event

        Parameters
        ----------
        name: str
            name of the model to plot
        **kwargs: Any kwargs you would pass to corner.corner

        Returns
        -------
        Fig handling
        """
        vars=self.binary_vars
        toplot = _np.column_stack([self.binary_params[var]['values'][self.detlabels] for var in vars])
        labels = [self.binary_params[var]['tex_label'] for var in vars]
        return _corner.corner(toplot,labels=labels,**kwargs)

    def histogram_astrophysical_population(self,**kwargs):
        """
        This method histogram the astrophysical distribution of GW event

        Parameters
        ----------
        name: str
            name of the model to plot
        **kwargs: Any kwargs you would pass to corner.corner

        Returns
        -------
        Fig handling
        """
        vars=self.binary_vars
        toplot = _np.column_stack([self.binary_params[var]['values'] for var in vars])
        labels = [self.binary_params[var]['tex_label'] for var in vars]
        return _corner.corner(toplot,labels=labels,**kwargs)

    def _histogramdn(self,bins_dict,**kwargs):
        """
        This method histogram the astrophysical distribution of GW event

        Parameters
        ----------
        bins_dict: dictionary
            Dictionary with in each filed the unmber of bins for each parameter
        **kwargs: Any kwargs you would pass to np.histogramdd

        Returns
        -------
        Ndimensional histogram from np.histogramdd, list of bin edges
        """
        vars=self.binary_vars
        bins=[bins_dict[var] for var in vars]
        toplot = _np.column_stack([self.binary_params[var]['values'] for var in vars])
        return _np.histogramdd(toplot,density=True,bins=bins,**kwargs)
