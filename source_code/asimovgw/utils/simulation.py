from pycbc.detector import Detector as  _Detector
from pycbc.psd.read import from_txt as _from_txt
from pycbc.waveform import get_fd_waveform as _get_fd_waveform
from pycbc.conversions import f_schwarzchild_isco as _f_schwarzchild_isco
import pycbc as _pycbc
import numpy as _np
from tqdm import tqdm as _tqdm
from .conversion import calculate_chis as _calculate_chis


"""
example

runs_dict={'O3':{'observing_time':0.5,'ifos':
                {'H1':{'duty_cycle':0.6,'asd_file':'asd/aligo_O3actual_H1.txt'},
                'L1':{'duty_cycle':0.7,'asd_file':'asd/aligo_O3actual_L1.txt'},
                'V1':{'duty_cycle':0.9,'asd_file':'asd/avirgo_O3actual.txt'}}} ,
          'O4':{'observing_time':1.0,'ifos':
                {'H1':{'duty_cycle':0.8,'asd_file':'asd/aligo_O4high.txt'},
                'L1':{'duty_cycle':0.8,'asd_file':'asd/aligo_O4high.txt'},
                'V1':{'duty_cycle':0.9,'asd_file':'asd/avirgo_O4high_NEW.txt'}}}
          }

params=[{'intrinsic':{'mass1':150.,'mass2':30.,
       'spin1x':0.,'spin1y':0.5, 'spin1z':0.,
       'spin2x':0.5,'spin2y':0., 'spin2z':0.,
       'coa_phase':0.,
       'inclination':0.,'distance':500},
       'extrinsic':{'right_ascension':0.,'declination':1.,'polarization':2.,'t_gps':1000000000}},
       {'intrinsic':{'mass1':150.,'mass2':30.,
       'spin1x':0.,'spin1y':0.5, 'spin1z':0.,
       'spin2x':0.5,'spin2y':0., 'spin2z':0.,
       'coa_phase':0.,
       'inclination':0.,'distance':500},
       'extrinsic':{'right_ascension':0.,'declination':1.,'polarization':2.,'t_gps':1000000000}}]

calculate_SNR(runs_dict,params,waveform_approximant='IMRPhenomXPHM',flow=20,delta_f=1)

"""

def calculate_SNR(runs_dict,list_params_dict,waveform_approximant,flow,delta_f,sampling_rate=4096.,verbose=True):

    tot_injections=len(list_params_dict)

    for i in _tqdm(range(len(list_params_dict)),desc='Calculating SNR for binary',disable=(not verbose)):
        params_dict_int=list_params_dict[i]['intrinsic']
        params_dict_ext=list_params_dict[i]['extrinsic']

        fhigh=0.5*sampling_rate

        hp,hc=_get_fd_waveform(approximant=waveform_approximant
                                             ,f_lower=flow,f_final=fhigh,delta_f=delta_f,**params_dict_int)

        obsper=_np.array([runs_dict[run]['observing_time'] for run in runs_dict.keys()])
        runs_available=[run for run in runs_dict.keys()]
        happeningin=_np.random.choice(runs_available,size=1,p=obsper/obsper.sum())[0]
        list_params_dict[i]['run']=happeningin
        ifo_dict=runs_dict[happeningin]['ifos']

        snrsq=[]
        snrsq_noise=[]
        turnon=[]
        for k,ifo in enumerate(ifo_dict.keys()):
            if _np.random.rand()<=ifo_dict[ifo]['duty_cycle']:
                turnon.append(ifo)
                det=_Detector(ifo)
                psd=_from_txt(filename=ifo_dict[ifo]['asd_file'],length=len(hp),
                                                   delta_f=delta_f,low_freq_cutoff=flow)

                Fp,Fc=det.antenna_pattern(**params_dict_ext)
                snrsq.append(_pycbc.filter.matchedfilter.sigmasq(htilde=Fp*hp+hc*Fc,
                                                             psd=psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh))
                snrsq_noise.append(_np.power(_np.sqrt(snrsq[k])+_np.random.randn(),2))
            else:
                snrsq.append(0.)
                snrsq_noise.append(0.)
            list_params_dict[i]['optimal_SNR2_IFO_{:s}'.format(ifo)]=snrsq[k]
            list_params_dict[i]['matched_filter_SNR2_IFO_{:s}'.format(ifo)]=snrsq_noise[k]

        list_params_dict[i]['ifo_on']=turnon
        list_params_dict[i]['optimal_SNR']=_np.sqrt(_np.sum(snrsq))
        list_params_dict[i]['matched_filter_SNR']=_np.sqrt(_np.sum(snrsq_noise))

    return list_params_dict



def generate_binaries(mp,zp,R0,Tobs,spins_mode,amax=0.9,Ngen=50000):


    redshift=zp.sample(Ngen)
    m1s,m2s=mp.sample(Ngen)
    mass1=m1s*(1.+redshift)
    mass2=m2s*(1.+redshift)

    if spins_mode=='aligned':
        a1=_np.random.uniform(low=0,high=amax,size=Ngen)
        a2=_np.random.uniform(low=0,high=amax,size=Ngen)
        cost1=_np.random.uniform(low=-1.,high=1.,size=Ngen)
        cost2=_np.random.uniform(low=-1.,high=1.,size=Ngen)
        cost1=_np.sign(cost1)*_np.ceil(_np.abs(cost1))
        cost2=_np.sign(cost2)*_np.ceil(_np.abs(cost2))

        spin1z,spin2z=a1*cost1,a2*cost2
        spin1x, spin2x = _np.zeros(Ngen),_np.zeros(Ngen)
        spin1y,spin2y=_np.zeros(Ngen),_np.zeros(Ngen)

    elif spins_mode=='isotropic':
        a1=_np.random.uniform(low=0,high=amax,size=Ngen)
        a2=_np.random.uniform(low=0,high=amax,size=Ngen)
        cost1=_np.random.uniform(low=-1.,high=1.,size=Ngen)
        cost2=_np.random.uniform(low=-1.,high=1.,size=Ngen)
        phixy1=_np.random.uniform(low=0.,high=_np.pi*2,size=Ngen)
        phixy2=_np.random.uniform(low=0.,high=_np.pi*2,size=Ngen)

        spin1z,spin2z=a1*cost1,a2*cost2
        spin1x, spin2x = a1*_np.sin(_np.arccos(cost1))*_np.cos(phixy1), a2*_np.sin(_np.arccos(cost2))*_np.cos(phixy2)
        spin1y,spin2y=  a1*_np.sin(_np.arccos(cost1))*_np.sin(phixy1), a2*_np.sin(_np.arccos(cost2))*_np.sin(phixy2)

    else:
        raise ValueError('spin distro Not known')

    chieff,chip=_calculate_chis(a1,a2,mass1,mass2,cost1,cost2)

    if spins_mode=='aligned':
        chip=_np.zeros_like(chieff)

    params=[]

    for i in _tqdm(range(Ngen),desc='Drawing binary'):


        params.append({'intrinsic':{'mass1':mass1[i],'mass2':mass2[i],
       'spin1x':spin1x[i],'spin1y':spin1y[i], 'spin1z':spin1z[i],
       'spin2x':spin2x[i],'spin2y':spin2y[i], 'spin2z':spin2z[i],
       'coa_phase':_np.random.uniform(low=0.,high=_np.pi*2,size=1)[0],
       'inclination':_np.arccos(_np.random.uniform(low=-1,high=1,size=1)[0]),'distance':zp.cosmo.dl_at_z(redshift[i])},
       'extrinsic':{'right_ascension':_np.random.uniform(low=0.,high=_np.pi*2,size=1)[0],
       'declination':_np.arcsin(_np.random.uniform(low=-1,high=1,size=1)[0]),'polarization':_np.random.uniform(low=0.,high=_np.pi*2,size=1)[0]
       ,'t_gps':1000000000+int(_np.random.uniform(low=0,high=3.14e7,size=1)[0]*Tobs)},
       'source':{'mass_1_source':m1s[i],'mass_2_source':m2s[i],'redshift':redshift[i],'chi_eff':chieff[i],'chi_p':chip[i]}
       })

    zlim=_np.linspace(0.,zp.cosmo.zmax,50000)
    Ntotal=_np.trapz(zp.prob_astro(zlim),zlim)*Tobs*R0
    return {'binaries':params,'Ntotal':int(Ntotal)}
