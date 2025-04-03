ppE.WaveformGeneratorPPE
========================================

.. code-block:: python

   class ppE.WaveformGeneratorPPE(duration=None,sampling_frequency=None,start_time=0,
                                                  frequency_domain_source_model=None,
                                                  time_domain_source_model=None,parameters=None,
                                                  parameter_conversion=None,waveform_arguments=None)

Bases: ``object``

Modified WaveformGenerator object from `bilby.gw.WaveformGenerator <https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.waveform_generator.WaveformGenerator.html#bilby.gw.waveform_generator.WaveformGenerator>`_ to include ppE corrections in the strain calculation.

To sample the ppE correction parameters, include `beta` (or `beta_tilde`), `delta_epsilon` (or `delta_epsilon_tilde`), and `b` in the prior.

.. code-block:: python

   __init__(duration=None,sampling_frequency=None,start_time=0,frequency_domain_source_model=None,
                                            time_domain_source_model=None,parameters=None,
                                            parameter_conversion=None,waveform_arguments=None)
