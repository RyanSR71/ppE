"ppE package"
__version__ = "0.1.2"

import numpy as np
from numpy.linalg import inv
import bilby
import random
import time as tm
import sys
import scipy
from bilby.core import utils
from bilby.core.series import CoupledTimeAndFrequencySeries
from bilby.core.utils import PropertyAccessor
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
import lal
from . import utils



class WaveformGeneratorPPE(object):
    duration = PropertyAccessor('_times_and_frequencies', 'duration')
    sampling_frequency = PropertyAccessor('_times_and_frequencies', 'sampling_frequency')
    start_time = PropertyAccessor('_times_and_frequencies', 'start_time')
    frequency_array = PropertyAccessor('_times_and_frequencies', 'frequency_array')
    time_array = PropertyAccessor('_times_and_frequencies', 'time_array')
    def __init__(self, duration=None, sampling_frequency=None, start_time=0, frequency_domain_source_model=None,
                 time_domain_source_model=None, parameters=None,
                 waveform_uncertainty_nodes=None,dA_sampling=False,dphi_sampling=False,phi_indexes=None,
                 parameter_conversion=None,
                 waveform_arguments=None):
        self._times_and_frequencies = CoupledTimeAndFrequencySeries(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    start_time=start_time)
        self.frequency_domain_source_model = frequency_domain_source_model
        self.time_domain_source_model = time_domain_source_model
        self.source_parameter_keys = self.__parameters_from_source_model()
        self.dA_sampling = dA_sampling
        self.dphi_sampling=dphi_sampling
        self.phi_indexes=phi_indexes
        
        if parameter_conversion is None:
            self.parameter_conversion = convert_to_lal_binary_black_hole_parameters
        else:
            self.parameter_conversion = parameter_conversion
        if waveform_arguments is not None:
            self.waveform_arguments = waveform_arguments
        else:
            self.waveform_arguments = dict()
        if isinstance(parameters, dict):
            self.parameters = parameters
        self._cache = dict(parameters=None, waveform=None, model=None)
        
        utils.logger.info(
            "Waveform generator initiated with\n"
            "  frequency_domain_source_model: {}\n"
            "  time_domain_source_model: {}\n"
            "  parameter_conversion: {}"
            .format(utils.get_function_path(self.frequency_domain_source_model),
                    utils.get_function_path(self.time_domain_source_model),
                    utils.get_function_path(self.parameter_conversion))
        )
        
    def __repr__(self):
        if self.frequency_domain_source_model is not None:
            fdsm_name = self.frequency_domain_source_model.__name__
        else:
            fdsm_name = None
        if self.time_domain_source_model is not None:
            tdsm_name = self.time_domain_source_model.__name__
        else:
            tdsm_name = None
        if self.parameter_conversion is None:
            param_conv_name = None
        else:
            param_conv_name = self.parameter_conversion.__name__
        return self.__class__.__name__ + '(duration={}, sampling_frequency={}, start_time={}, ' \
                                         'frequency_domain_source_model={}, time_domain_source_model={}, ' \
                                         'parameter_conversion={}, ' \
                                         'waveform_arguments={})'\
            .format(self.duration, self.sampling_frequency, self.start_time, fdsm_name, tdsm_name,
                    param_conv_name, self.waveform_uncertainty_nodes, self.dA_sampling, self.dphi_sampling, self.phi_indexes, self.waveform_arguments)
    
    def frequency_domain_strain(self, parameters=None):
        return self._calculate_strain(model=self.frequency_domain_source_model,
                                      model_data_points=self.frequency_array,
                                      parameters=parameters,
                                      transformation_function=utils.nfft,
                                      transformed_model=self.time_domain_source_model,
                                      transformed_model_data_points=self.time_array,
                                      )

    '''
    New time_domain_strain() function
    '''
    def time_domain_strain(self,parameters=None):
        fd_model_strain = self._calculate_strain(model=self.frequency_domain_source_model,
                                      model_data_points=self.frequency_array,
                                      parameters=parameters,
                                      transformation_function=utils.nfft,
                                      transformed_model=self.time_domain_source_model,
                                      transformed_model_data_points=self.time_array,
                                      )
        
        plus_td_waveform = self.sampling_frequency*np.fft.ifft(fd_model_strain['plus'])
        plus_td_model_strain = np.interp(self.time_array,np.linspace(self.time_array[0],self.time_array[-1],len(plus_td_waveform)),plus_td_waveform)

        cross_td_waveform = self.sampling_frequency*np.fft.ifft(fd_model_strain['cross'])
        cross_td_model_strain = np.interp(self.time_array,np.linspace(self.time_array[0],self.time_array[-1],len(cross_td_waveform)),cross_td_waveform)

        model_strain = dict()
        model_strain['plus'] = plus_td_model_strain
        model_strain['cross'] = cross_td_model_strain

        return model_strain
    
    def _calculate_strain(self, model, model_data_points, transformation_function, transformed_model,
                          transformed_model_data_points, parameters):
        if parameters is not None:
            self.parameters = parameters
        if self.parameters == self._cache['parameters'] and self._cache['model'] == model and \
                self._cache['transformed_model'] == transformed_model:
            return self._cache['waveform']
        if model is not None:
            model_strain = self._strain_from_model(model_data_points, model)
        elif transformed_model is not None:
            model_strain = self._strain_from_transformed_model(transformed_model_data_points, transformed_model,
                                                               transformation_function)
        else:
            raise RuntimeError("No source model given")
        self._cache['waveform'] = model_strain
        self._cache['parameters'] = self.parameters.copy()
        self._cache['model'] = model
        self._cache['transformed_model'] = transformed_model
        
        '''
        The following block performs the ppE correction:
        '''
        
        try:
            beta_tilde = parameters['beta_tilde']
            beta = utils.beta_from_beta_tilde_wrapped(beta_tilde,self.waveform_arguments['f_low'],1/np.pi,parameters['b'],0.018,total_mass)
        except:
            beta = 0
        
        '''
        try:
            beta = parameters['beta']
        except:
            beta = 0
    
        try:
            delta_epsilon = parameters['delta_epsilon']
        except:
            delta_epsilon = 0
        '''
        
        try:
            delta_epsilon_tilde = parameters['delta_epsilon_tilde']
            delta_epsilon = utils.beta_from_beta_tilde_wrapped(delta_epsilon_tilde,self.waveform_arguments['f_low'],1/np.pi,parameters['b'],0.018,total_mass)
        except:
            delta_epsilon = 0
        
        total_mass = bilby.gw.conversion.generate_mass_parameters(parameters)['total_mass']
               
        model_strain['plus'] = utils.apply_ppe_correction(model_strain['plus'],self.frequency_array,total_mass,beta,parameters['b'],delta_epsilon,0.018,0.5,True)
        model_strain['cross'] = utils.apply_ppe_correction(model_strain['cross'],self.frequency_array,total_mass,beta,parameters['b'],delta_epsilon,0.018,0.5,True)
            
        return model_strain
    
    def _strain_from_model(self, model_data_points, model):
        return model(model_data_points, **self.parameters)
    
    def _strain_from_transformed_model(self, transformed_model_data_points, transformed_model, transformation_function):
        transformed_model_strain = self._strain_from_model(transformed_model_data_points, transformed_model)
        if isinstance(transformed_model_strain, np.ndarray):
            return transformation_function(transformed_model_strain, self.sampling_frequency)
        model_strain = dict()
        for key in transformed_model_strain:
            if transformation_function == utils.nfft:
                model_strain[key], _ = \
                    transformation_function(transformed_model_strain[key], self.sampling_frequency)
            else:
                model_strain[key] = transformation_function(transformed_model_strain[key], self.sampling_frequency)
        return model_strain
    
    @property
    def parameters(self):
        return self.__parameters
    
    @parameters.setter 
    def parameters(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError('"parameters" must be a dictionary.')
        new_parameters = parameters.copy()
        new_parameters, _ = self.parameter_conversion(new_parameters)
        for key in self.source_parameter_keys.symmetric_difference(new_parameters):
            #############################################################################
            if key not in ['beta','delta_epsilon','beta_tilde','delta_epsilon_tilde','b']:  
                new_parameters.pop(key)
            #############################################################################
        self.__parameters = new_parameters
        self.__parameters.update(self.waveform_arguments)
        
    def __parameters_from_source_model(self):
        if self.frequency_domain_source_model is not None:
            model = self.frequency_domain_source_model
        elif self.time_domain_source_model is not None:
            model = self.time_domain_source_model
        else:
            raise AttributeError('Either time or frequency domain source '
                                 'model must be provided.')
        return set(utils.infer_parameters_from_function(model))
