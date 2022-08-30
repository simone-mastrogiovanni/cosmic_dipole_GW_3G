import bilby as _bilby
import sys as _sys
import os as _os

__all__ = ['initialize_prior_dict','initialize_events_dict',
'launch_analysis','write_condor_files','initialize_injections']

def write_condor_files(home_folder,uname='simone.mastrogiovanni',
agroup='ligo.dev.o3.cbc.hubble.gwcosmo',memory=None,cpus=None):
    """
    This function looks for all the *.py files in a folder and write a set of condor files
    needed for submission on write_condor_files. To launch the jobs, 1) Generate files with this function
    2) run chmod +x *.sh 3) launch the sub files.

    Parameters
    ----------
    home_folder: str
        Folder where to look for python files
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    """
    list_py_files = _os.listdir(home_folder)

    for file in list_py_files:
        if file.endswith('.py'):
            fname = file[:-3:1]

            f = open(home_folder+fname+'.sh', 'w')
            f.write('#!/bin/bash')
            f.write('\n')
            f.write('MYJOB_DIR='+home_folder)
            f.write('\n')
            f.write('cd ${MYJOB_DIR}')
            f.write('\n')
            f.write('python '+file)
            f.close()

            f = open(home_folder+fname+'.sub', 'w')
            f.write('universe = vanilla\n')
            f.write('getenv = True\n')
            f.write('executable = '+home_folder+fname+'.sh\n')
            f.write('accounting_group = '+agroup+'\n')
            f.write('accounting_group_user = '+uname)
            f.write('\n')
            if memory is not None:
                f.write('request_memory ='+str(memory)+'\n')
            if cpus is not None:
                f.write('request_cpus ='+str(cpus)+'\n')
            f.write('output = '+home_folder+fname+'.stdout\n')
            f.write('error = '+home_folder+fname+'.stderr\n')
            f.write('queue\n')
            f.close()


def initialize_prior_dict(population,cosmology,rates,scale_free):
    '''
    This function initialize a prior dictionary to run bilby jobs with the hierarchical population anlysis

    Parameters
    ----------
    population: str
        The source-frame mass population model, either 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw' or 'BBH-powerlaw-double-gaussian'
    cosmology: str
        The cosmological model, either 'flatLCDM', 'w0flatLCDM', 'w0waflatLCDM'
    rates: str
        The merger rate evolution model, either 'powerlaw' or 'madau'
    scale_free: boolean
        If false it will provide a prior also on the rate of merger today.

    Returns
    -------
    prior_dict: dict
        The prior dictionary
    '''
    prior_dict = {}

    if rates=='non-evolving':
        prior_dict['gamma'] = _bilby.core.prior.DeltaFunction(0,name='gamma')
    elif rates=='powerlaw':
        prior_dict['gamma'] = _bilby.core.prior.Uniform(-4,10,name='gamma')
    elif rates=='madau':
        prior_dict['gamma'] = _bilby.core.prior.Uniform(-4,10,name='gamma')
        prior_dict['kappa'] = _bilby.core.prior.Uniform(-6,6,name='kappa')
        prior_dict['zp'] = _bilby.core.prior.Uniform(0,5,name='zp')
    else:
        raise ValueError('Redshift model not known')

    if not scale_free:
        prior_dict['R0'] = _bilby.core.prior.Uniform(0,1000,name='R0')

    if population == 'BBH-powerlaw':
        prior_dict['alpha'] = _bilby.core.prior.Uniform(-4,12,name='alpha')
        prior_dict['beta'] = _bilby.core.prior.Uniform(-4,12,name='beta')
        prior_dict['mmax'] = _bilby.core.prior.Uniform(30,100,name='mmax')
        prior_dict['mmin'] = _bilby.core.prior.Uniform(2,10,name='mmin')

    elif population == 'fixed-BBH-powerlaw':
        prior_dict['alpha'] = _bilby.core.prior.DeltaFunction(2,name='alpha')
        prior_dict['beta'] = _bilby.core.prior.DeltaFunction(0,name='beta')
        prior_dict['mmax'] = _bilby.core.prior.DeltaFunction(50,name='mmax')
        prior_dict['mmin'] = _bilby.core.prior.DeltaFunction(5,name='mmin')

    elif population == 'BBH-powerlaw-gaussian':

        prior_dict['alpha'] = _bilby.core.prior.Uniform(-4,12,name='alpha')
        prior_dict['beta'] = _bilby.core.prior.Uniform(-4,12,name='beta')
        prior_dict['mmax'] = _bilby.core.prior.Uniform(30,100,name='mmax')
        prior_dict['mmin'] = _bilby.core.prior.Uniform(2,10,name='mmin')
        prior_dict['mu_g'] = _bilby.core.prior.Uniform(20,50,name='mu_g')
        prior_dict['sigma_g'] = _bilby.core.prior.Uniform(0.4,10,name='sigma_g')
        prior_dict['lambda_peak'] = _bilby.core.prior.Uniform(0,1,name='lambda_peak')
        prior_dict['delta_m'] = _bilby.core.prior.Uniform(0,10,name='delta_m')

    elif population == 'fixed-BBH-powerlaw-gaussian':
        prior_dict['alpha'] = _bilby.core.prior.DeltaFunction(2,name='alpha')
        prior_dict['beta'] = _bilby.core.prior.DeltaFunction(0,name='beta')
        prior_dict['mmax'] = _bilby.core.prior.DeltaFunction(50,name='mmax')
        prior_dict['mmin'] = _bilby.core.prior.DeltaFunction(5,name='mmin')
        prior_dict['mu_g'] = _bilby.core.prior.DeltaFunction(35,name='mu_g')
        prior_dict['sigma_g'] = _bilby.core.prior.DeltaFunction(5,name='sigma_g')
        prior_dict['lambda_peak'] = _bilby.core.prior.DeltaFunction(0.5,name='lambda_peak')
        prior_dict['delta_m'] = _bilby.core.prior.DeltaFunction(2,name='delta_m')

    elif population == 'BBH-broken-powerlaw':

        prior_dict['alpha_1'] = _bilby.core.prior.Uniform(-4,12,name='alpha_1')
        prior_dict['alpha_2'] = _bilby.core.prior.Uniform(-4,12,name='alpha_2')
        prior_dict['beta'] = _bilby.core.prior.Uniform(-4,12,name='beta')
        prior_dict['mmax'] = _bilby.core.prior.Uniform(30,100,name='mmax')
        prior_dict['mmin'] = _bilby.core.prior.Uniform(2,10,name='mmin')
        prior_dict['b'] = _bilby.core.prior.Uniform(0,1,name='b')
        prior_dict['delta_m'] = _bilby.core.prior.Uniform(0,10,name='delta_m')

    elif population == 'fixed-BBH-broken-powerlaw':

        prior_dict['alpha_1'] = _bilby.core.prior.DeltaFunction(2,name='alpha_1')
        prior_dict['alpha_2'] = _bilby.core.prior.DeltaFunction(1,name='alpha_2')
        prior_dict['beta'] = _bilby.core.prior.DeltaFunction(0,name='beta')
        prior_dict['mmax'] = _bilby.core.prior.DeltaFunction(50,name='mmax')
        prior_dict['mmin'] = _bilby.core.prior.DeltaFunction(5,name='mmin')
        prior_dict['b'] = _bilby.core.prior.DeltaFunction(0.5,name='b')
        prior_dict['delta_m'] = _bilby.core.prior.DeltaFunction(2,name='delta_m')

    elif population == 'BBH-powerlaw-double-gaussian':
        prior_dict['alpha'] = _bilby.core.prior.Uniform(-4,12,name='alpha')
        prior_dict['beta'] = _bilby.core.prior.Uniform(-4,12,name='beta')
        prior_dict['mmax'] = _bilby.core.prior.Uniform(30,100,name='mmax')
        prior_dict['mmin'] = _bilby.core.prior.Uniform(2,10,name='mmin')

        prior_dict['mu_g_low'] = _bilby.core.prior.Uniform(20,50,name='mu_g_low')
        prior_dict['sigma_g_low'] = _bilby.core.prior.Uniform(0.4,10,name='sigma_g_low')
        prior_dict['mu_g_high'] = _bilby.core.prior.Uniform(50,100,name='mu_g_high')
        prior_dict['sigma_g_high'] = _bilby.core.prior.Uniform(0.4,10,name='sigma_g_high')

        prior_dict['lambda_g'] = _bilby.core.prior.Uniform(0,1,name='lambda_g')
        prior_dict['lambda_g_low'] = _bilby.core.prior.Uniform(0,1,name='lambda_g_low')

        prior_dict['delta_m'] = _bilby.core.prior.Uniform(0,10,name='delta_m')

    else:
        print('Prior model not implemented')
        _sys.exit()

    if cosmology == 'fixed-flatLCDM':
        prior_dict['H0'] = _bilby.core.prior.DeltaFunction(67.74,name='H0')
        prior_dict['Om0'] = _bilby.core.prior.DeltaFunction(0.3065,name='Om0')

    elif cosmology == 'flatLCDM':
        prior_dict['H0'] = _bilby.core.prior.Uniform(20,120,name='H0')
        prior_dict['Om0'] = _bilby.core.prior.Uniform(0.1,0.5,name='Om0')
    elif cosmology == 'w0flatLCDM':
        prior_dict['H0'] = _bilby.core.prior.Uniform(20,120,name='H0')
        prior_dict['Om0'] = _bilby.core.prior.Uniform(0.1,0.5,name='Om0')
        prior_dict['w0'] = _bilby.core.prior.Uniform(-3.0,0.,name='w0')

    elif cosmology == 'w0waflatLCDM':
        prior_dict['H0'] = _bilby.core.prior.Uniform(20,120,name='H0')
        prior_dict['Om0'] = _bilby.core.prior.Uniform(0.1,0.5,name='Om0')
        prior_dict['w0'] = _bilby.core.prior.Uniform(-3.0,0.,name='w0')
        prior_dict['wa'] = _bilby.core.prior.Uniform(-2.,2.,name='wa')

    elif cosmology == 'restricted-flatLCDM':
        prior_dict['H0'] = _bilby.core.prior.Uniform(65.,77.,name='H0')
        prior_dict['Om0'] = _bilby.core.prior.DeltaFunction(0.3065,name='Om0')
    else:
        print('Not implemented')
        _sys.exit()

    return prior_dict

def initialize_events_dict(runs,type=None,snr_cut=None,ifar_cut=None,to_include = None):
    '''
    This function returns a dictionary of GW events given some criteria. Useful when you want to select events without
    wasting too much time

    Parameters
    ----------
    runs: list
        List of runs to include, e.g. ['O1','O2','O3a','O3b']
    type: list
        Type of event,'BBH','BNS' or 'NSBH'
    snr_cut: float
        SNR cut to apply (select above)
    ifar_cut: float
        ifar cut to apply (select above)
    to_include: list
        List of GW names to include regardless of the criteria.
    '''


    # Runs in ['O1','O2','O3a']

    if snr_cut is None:
        snr_cut=0
    if ifar_cut is None:
        ifar_cut=0


    # This table is build from https://arxiv.org/abs/1811.12907

    events_dict= {'GW150914': {'SNR': 24.4, 'ifar': 1/1.0e-7,'RUN':'O1','TYPE':'BBH','IFOS':'HL'} ,
    'GW151012': {'SNR': 10.0, 'ifar': 1/7.9e-3,'RUN':'O1','TYPE':'BBH','IFOS':'HL'},
    'GW151226': {'SNR': 13.1, 'ifar': 1/1.0e-7,'RUN':'O1','TYPE':'BBH','IFOS':'HL'},
    'GW170104': {'SNR': 13.0, 'ifar': 1/1.0e-7,'RUN':'O2','TYPE':'BBH','IFOS':'HL'},
    'GW170608': {'SNR': 15.4, 'ifar': 1/1.0e-7,'RUN':'O2','TYPE':'BBH','IFOS':'HL'},
    'GW170729': {'SNR': 10.8, 'ifar': 1/0.18,'RUN':'O2','TYPE':'BBH','IFOS':'HLV'},
    'GW170809': {'SNR': 12.4, 'ifar': 1/1.0e-7,'RUN':'O2','TYPE':'BBH','IFOS':'HLV'},
    'GW170814': {'SNR': 16.3, 'ifar': 1/1.0e-7,'RUN':'O2','TYPE':'BBH','IFOS':'HLV'},
    'GW170817': {'SNR': 33.0, 'ifar': 1/1.0e-7,'RUN':'O2','TYPE':'BNS','IFOS':'HLV'},
    'GW170818': {'SNR': 11.3, 'ifar': 1/4.2e-5,'RUN':'O2','TYPE':'BBH','IFOS':'HLV'},
    'GW170823': {'SNR': 11.5, 'ifar': 1/1.0e-7,'RUN':'O2','TYPE':'BBH','IFOS':'HLV'},

    # Taken from the O3b catalog paper

    'GW190403_051519': {'SNR': 7.96, 'ifar': 1.294399e-01,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190403A'},
    'GW190408_181802': {'T_NAME':'S191103a','SNR': 14.79, 'ifar': 1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190408A'},
    'GW190412': {'T_NAME':'S190412m','SNR': 19.67, 'ifar': 1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190412A'},
    'GW190413_052954': {'T_NAME':'S190413i','SNR': 8.50, 'ifar': 1.221589e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190413A'},
    'GW190413_134308': {'T_NAME':'S190413ac','SNR': 10.29, 'ifar': 5.536435e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190413B'},
    'GW190421_213856': {'T_NAME':'S190421ar','SNR': 10.49, 'ifar': 3.536528e+02,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190421A'},
    'GW190424_180648': {'T_NAME':'S190424ao','SNR': 10.0, 'ifar': 1/2,'RUN':'O3a','TYPE':'BBH','IFOS':'L','PE_NAME':'GW190424A'}, # Set threshold of 1/2yr according to GWTC2.1, SNR from GWTC-2
    'GW190425': {'T_NAME':'S190425z','SNR': 12.86, 'ifar': 2.957851e+01,'RUN':'O3a','TYPE':'BNS','IFOS':'LV','PE_NAME':'GW190425A'},
    'GW190426_152155': {'T_NAME':'-','SNR': 10.1, 'ifar': 1/0.91,'RUN':'O3a','TYPE':'NSBH','IFOS':'HLV','PE_NAME':'GW190426A'}, # Compatible with NSB but high FAR, taken from Tab. III of GWTC-2.1
    'GW190426_190642': {'T_NAME':'S190426c','SNR': 9.57, 'ifar': 2.450886e-01,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190426B'}, # New massive event in GWTC2.1
    'GW190503_185404': {'T_NAME':'S190503bf','SNR': 12.80, 'ifar': 1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190503A'},
    'GW190512_180714': {'T_NAME':'S190512at','SNR':12.37,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190512A'},
    'GW190513_205428': {'T_NAME':'S190513bm','SNR':12.95,'ifar':7.533496e+04,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190513A'},
    'GW190514_065416': {'T_NAME':'S190514n','SNR':8.40,'ifar':3.634077e-01,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190514A'},
    'GW190517_055101': {'T_NAME':'S190517h','SNR':11.32,'ifar':2.892024e+03,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190517A'},
    'GW190519_153544': {'T_NAME':'S190519bj','SNR':13.95,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190519A'},
    'GW190521': {'T_NAME':'S190521g','SNR':14.38,'ifar':4.900104e+03,'RUN':'O3a','TYPE':'SPECIAL','IFOS':'HLV','PE_NAME':'GW190521A'}, # Classified as special because very massive
    'GW190521_074359': {'T_NAME':'S190521r','SNR':24.67,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190521B'},
    'GW190527_092055': {'T_NAME':'S190527w','SNR':8.70,'ifar':4.376144e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190527A'},
    'GW190531_023648': {'T_NAME':'-','SNR':10.0,'ifar':1/0.41,'RUN':'O3a','TYPE':'NSBH','IFOS':'HLV','PE_NAME':'GW190531A'}, #New troublesome event, possibly NSBH from GWTC2.1, taken from GWTC-3
    'GW190602_175927': {'T_NAME':'S190602aq','SNR':12.60,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190602A'},
    'GW190620_030421': {'T_NAME':'S190620e','SNR':10.87,'ifar':8.959435e+01,'RUN':'O3a','TYPE':'BBH','IFOS':'LV','PE_NAME':'GW190620A'},
    'GW190630_185205': {'T_NAME':'S190630ag','SNR':15.19,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'LV','PE_NAME':'GW190630A'},
    'GW190701_203306': {'T_NAME':'S190701ah','SNR':11.88,'ifar':1.751977e+02,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190701A'},
    'GW190706_222641': {'T_NAME':'S190706ai','SNR':12.65,'ifar':2.000340e+04,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190706A'},
    'GW190707_093326': {'T_NAME':'S190707q','SNR':13.19,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190707A'},
    'GW190708_232457': {'T_NAME':'S190708ap','SNR':13.12,'ifar':3.240890e+03,'RUN':'O3a','TYPE':'BBH','IFOS':'LV','PE_NAME':'GW190708A'},
    'GW190719_215514': {'T_NAME':'S190719an','SNR':7.98,'ifar':1.586365e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190719A'},
    'GW190720_000836': {'T_NAME':'S190720a','SNR':11.64,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190720A'},
    'GW190725_174728': {'T_NAME':'S190725t','SNR':9.84,'ifar':2.182329e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190725A'}, # One of the new events has some prob in pastro NSBH
    'GW190727_060333': {'T_NAME':'S190727h','SNR':12.05,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190727A'},
    'GW190728_064510': {'T_NAME':'S190728q','SNR':13.37,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190728A'},
    'GW190731_140936': {'T_NAME':'S190731aa','SNR':9.13,'ifar':2.989144e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190731A'},
    'GW190803_022701': {'T_NAME':'S190803e','SNR':9.06,'ifar':1.366321e+01,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190803A'},
    'GW190805_211137': {'T_NAME':'S190805bq','SNR':8.25,'ifar':1.594814e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190805A'},
    'GW190814': {'T_NAME':'S190814bv','SNR':22.18,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'SPECIAL','IFOS':'LV','PE_NAME':'GW190814A'},
    'GW190828_063405': {'T_NAME':'S190828j','SNR':16.63,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190828A'},
    'GW190828_065509': {'T_NAME':'S190828l','SNR':11.07,'ifar':2.874007e+04,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190828B'},
    'GW190909_114149': {'T_NAME':'S190909w','SNR': 8.5, 'ifar': 1/1.1,'RUN':'O3a','TYPE':'SPECIAL','IFOS':'HL','PE_NAME':'GW190909A'}, # Recovered even with smaller SNR in GWTC2.1, deranked, TAB IV
    'GW190910_112807': {'T_NAME':'S190910s','SNR':13.42,'ifar':3.486817e+02,'RUN':'O3a','TYPE':'BBH','IFOS':'LV','PE_NAME':'GW190910A'},
    'GW190915_235702': {'T_NAME':'S190915ak','SNR':13.07,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190915A'},
    'GW190916_200658': {'T_NAME':'-','SNR':8.22,'ifar':2.106446e-01,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190916A'}, # New event from GWTC2.1, a BBH
    'GW190917_114630': {'T_NAME':'S190917u','SNR':9.46,'ifar':1.523161e+00,'RUN':'O3a','TYPE':'NSBH','IFOS':'HLV','PE_NAME':'GW190917A'}, # New event in GWTC2.1, a possible NSBH like GW190814
    'GW190924_021846': {'T_NAME':'S190924h','SNR':12.96,'ifar':1.000000e+05,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190924A'},
    'GW190925_232845': {'T_NAME':'S190925ad','SNR':9.88,'ifar':1.387649e+02,'RUN':'O3a','TYPE':'BBH','IFOS':'HV','PE_NAME':'GW190925A'}, # New event from GWTC2.1, a BBH
    'GW190926_050336': {'T_NAME':'-','SNR':9.00,'ifar':8.707174e-01,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190926A'}, # New event from GWTC2.1, a BBH
    'GW190929_012149': {'T_NAME':'S190929d','SNR': 10.32, 'ifar': 6.450435e+00,'RUN':'O3a','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW190929A'},
    'GW190930_133541': {'T_NAME':'S190930s','SNR': 10.12, 'ifar': 8.096156e+01,'RUN':'O3a','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW190930A'},

    #O3b below. This table is built from the last version of the GWTC-3 paper https://pubplan.ligo.org/CBC/O3b_CBC_catalog.html

    'GW191103_012549': {'T_NAME':'S191103a','SNR':9.27,'ifar':2.182086e+00,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191103A'},
    'GW191105_143521': {'T_NAME':'S191105e','SNR':10.67,'ifar':8.445714e+01,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW191105A'},
    'GW191109_010717': {'T_NAME':'S191109d','SNR':15.78,'ifar':5.561309e+03,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191109A'},
    'GW191113_071753': {'T_NAME':'S191113q','SNR':9.21,'ifar':3.876905e-02,'RUN':'O3b','TYPE':'BBH','IFOS':'LV','PE_NAME':'GW191113A'},
    'GW191126_115259': {'T_NAME':'S191126l','SNR':8.65,'ifar':3.127243e-01,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191126A'},
    'GW191127_050227': {'T_NAME':'S191127p','SNR':10.30,'ifar':4.018647e+00,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW191127A'},
    'GW191129_134029': {'T_NAME':'S191129u','SNR':13.32,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191129A'},
    'GW191204_110529': {'T_NAME':'S191204h','SNR':9.03,'ifar':3.064248e-01,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191204A'},
    'GW191204_171526': {'T_NAME':'S191204r','SNR':17.12,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191204B'},
    'GW191213G': {'T_NAME':'S191213bb','SNR':10.02,'ifar':2.797793e-01,'RUN':'O3b','TYPE':'TRIGGER','IFOS':'HLV','PE_NAME':'-'},
    'GW191215_223052': {'T_NAME':'S191215w','SNR':10.91,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW191215A'},
    'GW191216_213338': {'T_NAME':'S191216ap','SNR':18.61,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HV','PE_NAME':'GW191216A'},
    'GW191219_163120': {'T_NAME':'S191219ax','SNR':8.91,'ifar':2.498278e-01,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'-'},
    'GW191222_033537': {'T_NAME':'S191222n','SNR':12.04,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191222A'},
    'GW191230_180458': {'T_NAME':'S191230an','SNR':10.28,'ifar':1.995610e+01,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW191230A'},
    'GW200105': {'T_NAME':'S200105ae','SNR':13.88,'ifar':4.908019e+00,'RUN':'O3b','TYPE':'NSBH','IFOS':'L','PE_NAME':'GW200105A'},
    'GW200112_155838': {'T_NAME':'S200112r','SNR':17.64,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'LV','PE_NAME':'GW200112A'},
    'GW200115': {'T_NAME':'S200115j','SNR': 11.47, 'ifar': 1.000000e+05,'RUN':'O3b','TYPE':'NSBH','IFOS':'HL','PE_NAME':'GW200115A'},
    'GW200128_022011': {'T_NAME':'S200128d','SNR':10.14,'ifar':2.328150e+02,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200128A'},
    'GW200129_065458': {'T_NAME':'S200129m','SNR':26.51,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW200129A'},
    'GW200202_154313': {'T_NAME':'S200202ac','SNR':11.32,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200202A'},
    'GW200208_130117': {'T_NAME':'S200208q','SNR':10.82,'ifar':3.218858e+03,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW200208A'},
    'GW200208_222617': {'T_NAME':'S200208am','SNR':8.91,'ifar':2.070506e-01,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200208B'},
    'GW200209_085452': {'T_NAME':'S200209ab','SNR':10.03,'ifar':2.162810e+01,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200209A'},
    'GW200210_092254': {'T_NAME':'S200210ba','SNR':9.55,'ifar':8.024908e-01,'RUN':'O3b','TYPE':'NSBH','IFOS':'HL','PE_NAME':'GW200210A'},
    'GW200216_220804': {'T_NAME':'S200216br','SNR':9.40,'ifar':2.865817e+00,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200216A'},
    'GW200218G': {'T_NAME':'S200218al', 'SNR':8.68,'ifar':4.634265e-01,'RUN':'O3b','TYPE':'TRIGGER','IFOS':'HL','PE_NAME':'-'},
    'GW200219_094415': {'T_NAME':'S200219ac','SNR':10.75,'ifar':1.005337e+03,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200219A'},
    'GW200220_061928': {'T_NAME':'S200220ad','SNR':7.50,'ifar':1.466872e-01,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW200220A'},
    'GW200220_124850': {'T_NAME':'S200220aw','SNR':8.25,'ifar':3.364207e-02,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200220B'},
    'GW200224_030524': {'T_NAME':'S200224o','SNR':8.74,'ifar':3.039129e-01,'RUN':'O3b','TYPE':'TRIGGER','IFOS':'HLV','PE_NAME':'-'},
    'GW200224_222234': {'T_NAME':'S200224ca','SNR':19.18,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW200224A'},
    'GW200225_060421': {'T_NAME':'S200225q','SNR':13.09,'ifar':8.764500e+04,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200225A'},
    'GW200302_015811': {'T_NAME':'S200302c','SNR':10.60,'ifar':8.955210e+00,'RUN':'O3b','TYPE':'BBH','IFOS':'H','PE_NAME':'GW200302A'},
    'GW200306_093714': {'T_NAME':'S200306ak','SNR':8.49,'ifar':4.254211e-02,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200306A'},
    'GW200308_173609': {'T_NAME':'S200308bl','SNR':8.26,'ifar':4.215515e-01,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200308A'},
    'GW200311_115853': {'T_NAME':'S200311bg','SNR':17.65,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW200311B'},
    'GW200316_215756': {'T_NAME':'S200316bj','SNR':10.08,'ifar':1.000000e+05,'RUN':'O3b','TYPE':'BBH','IFOS':'HL','PE_NAME':'GW200316A'},
    'GW200322_091133': {'T_NAME':'S200322ab','SNR':9.03,'ifar':7.171419e-03,'RUN':'O3b','TYPE':'BBH','IFOS':'HLV','PE_NAME':'GW200322A'},
    'GW200326_112501': {'T_NAME':'S200326af','SNR':9.16,'ifar':4.219194e-01,'RUN':'O3b','TYPE':'TRIGGER','IFOS':'HL','PE_NAME':'-'}
    }

    psd_events_out = {}


    for event in list(events_dict.keys()):
        if to_include is not None:
            if event in to_include:
                psd_events_out[event] = events_dict[event]
                continue

        if not (events_dict[event]['RUN'] in runs):
            continue

        if type is not None:
            if (events_dict[event]['TYPE'] in type) and (events_dict[event]['SNR'] >= snr_cut) and (events_dict[event]['ifar'] >= ifar_cut):
                psd_events_out[event] = events_dict[event]
        else:
            if (events_dict[event]['SNR'] >= snr_cut) and (events_dict[event]['ifar'] >= ifar_cut):
                psd_events_out[event] = events_dict[event]

    return psd_events_out
