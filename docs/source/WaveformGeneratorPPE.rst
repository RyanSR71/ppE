ppE.WaveformGeneratorPPE
========================================

.. code-block:: python

   class ppE.WaveformGeneratorPPE(duration=None,sampling_frequency=None,start_time=0,
                                                  frequency_domain_source_model=None,
                                                  time_domain_source_model=None,parameters=None,
                                                  parameter_conversion=None,waveform_arguments=None)

Bases: ``object``

Modified WaveformGenerator object from `bilby.gw.WaveformGenerator <https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.waveform_generator.WaveformGenerator.html#bilby.gw.waveform_generator.WaveformGenerator>`_ to include ppE corrections in the strain calculation.

.. math::

   \mu_\mathrm{ppE}(f;\theta,b,\beta,\delta\epsilon)=\mu(f;\theta)\exp[i\Delta\phi_\mathrm{ppE}(f;b,\beta,\delta\epsilon)]

.. math::

   \Delta\phi_\mathrm{ppE}(f;b,\beta,\delta\epsilon)=\begin{cases}
        \beta u^b & f<f_\mathrm{IM} \\
        \Delta\phi_\mathrm{Int}(f;\beta,\delta\epsilon) & f_\mathrm{IM}\leq f<\frac{1}{2}f_\mathrm{RD} \\
        \beta u_\mathrm{IM}^b+\frac{b}{3}\delta\epsilon u_\mathrm{IM}^b\left(\left(\frac{u}{u_\mathrm{IM}}\right)^3-1\right) & \frac{1}{2}f_\mathrm{RD}\leq f
    \end{cases}

To sample the ppE correction parameters, include `beta` (or `beta_tilde`), `delta_epsilon` (or `delta_epsilon_tilde`), and `b` in the prior.

.. code-block:: python

   __init__(duration=None,sampling_frequency=None,start_time=0,frequency_domain_source_model=None,
                                            time_domain_source_model=None,parameters=None,
                                            parameter_conversion=None,waveform_arguments=None)
