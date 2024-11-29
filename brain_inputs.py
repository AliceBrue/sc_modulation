""" functions to define sinusoidal brain inputs """

import numpy as np
from matplotlib import pyplot as plt

def brain_inputs(params_flex, params_ext, time):
    
    offset_flex, frequency_flex, amplitude_flex, phase_flex = params_flex
    offset_ext, frequency_ext, amplitude_ext, phase_ext = params_ext
    signal_flex = offset_flex + amplitude_flex * np.sin(2 * np.pi * frequency_flex * time + phase_flex)
    signal_delt = 0 * np.sin(2 * np.pi * frequency_flex * time + phase_flex)
    signal_ext = offset_ext+amplitude_ext * np.sin(2 * np.pi * frequency_ext * time + phase_ext)
    #signal_flex = (amplitude_flex/2) + (amplitude_flex/2) * np.sin(2 * np.pi * frequency_flex * time + phase_flex)
    #signal_delt = 0 * np.sin(2 * np.pi * frequency_flex * time + phase_flex)
    #signal_ext = (amplitude_ext/2) + (amplitude_ext/2) * np.sin(2 * np.pi * frequency_ext * time + phase_ext)
    controls_dict = {}
    controls_dict["signal_values_flex"] = list(signal_flex)
    controls_dict["signal_values_ext"] = list(signal_ext)
    controls_dict["signal_delt"]=list(signal_delt)
    
    return controls_dict
