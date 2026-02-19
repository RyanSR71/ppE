import os
import numpy as np
import bilby
import matplotlib.pyplot as plt
import lal
import pesummary
from pesummary.gw.file.strain import StrainData
from pesummary.io import read
import GWCorrect.ppE as ppE

from importlib.metadata import version
print(version('GWCorrect'))

for b in [-3,-1,1]:
    
    duration = 16
    
    injection = dict(
        chirp_mass=34.8,
        mass_ratio=1,
        chi_1=0,
        chi_2=0,
        luminosity_distance=1000,
        geocent_time=1126259642.5,
        phase=1.577,
        theta_jn=0.48736165,
        dec=-1.0108,
        ra=1.7475,
        psi=2.6597,
    )

    injection['beta_tilde'] = 0.5
    injection['delta_epsilon_tilde'] = -0.25
    injection['b'] = b

    prior = bilby.core.prior.PriorDict()

    for key in injection.keys():
        prior[key] = bilby.core.prior.DeltaFunction(name=key,peak=injection[key])

    #prior['total_mass'] = ppE.prior.TotalMassConstraintPPE(name='total_mass',latex_label=r'$M$',f_low = 20,unit=r'$\mathrm{M}_{\odot}$')
    prior['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass',latex_label=r'$\mathcal{M}_c$',minimum=5,maximum=50,unit=r'$\mathrm{M}_{\odot}$')
    #prior['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio',latex_label=r'$q$',minimum=0.125,maximum=1)
    #prior['chi_1'] = bilby.core.prior.Uniform(name='chi_1',latex_label=r'$\chi_1$',minimum=-1,maximum=1)
    #prior['chi_2'] = bilby.core.prior.Uniform(name='chi_2',latex_label=r'$\chi_2$',minimum=-1,maximum=1)
    prior['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance',latex_label=r'$D_L$',minimum=50,maximum=10000,unit='Mpc')
    prior['geocent_time'] = bilby.core.prior.Uniform(name='geocent_time',latex_label=r'$t_{c}$',minimum=1126259642.4,maximum=1126259642.6,unit='s')
    prior['phase'] = bilby.core.prior.Uniform(name='phase',latex_label=r'$\phi_\mathrm{ref}$',minimum=0,maximum=6.2831853071795865,boundary='periodic')

    prior['beta_tilde'] = bilby.core.prior.Uniform(name='beta_tilde',latex_label=r'$\tilde{\beta}$',minimum=-2,maximum=2)
    prior['delta_epsilon_tilde'] = bilby.core.prior.Uniform(name='delta_epsilon_tilde',latex_label=r'$\delta\tilde{\epsilon}$',minimum=-2,maximum=2)
    prior['b'] = bilby.core.prior.Uniform(name='b',latex_label=r'$b$',minimum=-5,maximum=4)

    waveform_arguments = dict(waveform_approximant='IMRPhenomD', reference_frequency=20.0, 
                              catch_waveform_errors=True, minimum_frequency=20.0, maximum_frequency=2048.0,
                              aligned=True)

    hf1 = bilby.gw.WaveformGenerator(parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                        waveform_arguments=waveform_arguments,
                        frequency_domain_source_model=ppE.waveform_generator.ppECorrectionModel,
                        sampling_frequency=4096, 
                        duration=duration
                    )

    ifos = bilby.gw.detector.InterferometerList(['H1','L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=4096, duration=duration,
        start_time=injection['geocent_time'] - 2)
    ifo_injection = ifos.inject_signal(
        waveform_generator=hf1,
        parameters=injection)

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifos,
        hf1,
        priors=prior,
        time_marginalization=False, 
        phase_marginalization=True, 
        distance_marginalization=True,
    )

    nlive = 1000
    dlogz = 0.01

    result = bilby.run_sampler(
        likelihood, prior, sampler='nestle', outdir='/home/ryanmatthew.johnson/bGR/output/paper_runs', 
        label=f"bGR-bGR_high-mass_H1-L1_b={b}_nlive{str(nlive)}_dlogz={dlogz}_v{version('GWCorrect')}",
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        nlive=nlive, 
        dlogz=dlogz,
        clean=True,
        maxiter=None,
        verbose=True,
    )