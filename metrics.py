""" functions to compute metrics """

import numpy as np
from scipy.signal import *
import os


def rmse(movement, target_mov):

    # interp target
    target = np.interp(np.linspace(0, 1, len(movement)), np.linspace(0, 1, len(target_mov)), target_mov)
    _rmse = np.sqrt(np.mean((movement - target)**2))

    return _rmse


def comp_emg(sim_folder, muscle_list_flex=['DELT1', 'BIClong', 'BICshort', 'BRA'], muscle_list_ext=['TRIlong', 'TRIlat'], w_neg = 0.5):

    if os.path.isfile(sim_folder + "activation_dict_av.txt"):

        res_file = open(sim_folder + "activation_dict_av.txt", "r")
        res_lines = res_file.readlines()
        for l in res_lines:
            if "extensor_overlap" in l:
                overlap_ext = float(l.split(": ")[-1])
            if "extensor_noverlap" in l:
                noverlap_ext = float(l.split(": ")[-1])
            if "flexor_overlap" in l:
                overlap_flex = float(l.split(": ")[-1])
            if "flexor_noverlap" in l:
                noverlap_flex = float(l.split(": ")[-1])

    else:
        print("RUN EMG_Activation.py for the corresponding scenartios first")

    return np.mean(overlap_flex) - w_neg*np.mean(noverlap_flex), np.mean(overlap_ext) - w_neg*np.mean(noverlap_ext)


def deviation(movement, perturb_time, step_size):

    _dev = np.max(movement) - movement[int(perturb_time/step_size)+1]

    return _dev


def range_motion(movement):

    _range = np.max(movement) - np.min(movement)

    return _range


def n_oscillation_peaks(movement, dist):

    peaks, _ = find_peaks(movement, distance=dist)
    metric = len(peaks)

    return metric


def damping(movement, time, peaks, w, h_peak): 

    l_peak = peaks[-1] 
    if peaks[0] > 0.75*h_peak:
        h_peak = peaks[0]
    damp_end = min(l_peak+int(w/2), len(movement)-1) 
    damp = np.abs((movement[h_peak] - movement[damp_end])/(time[h_peak] - time[damp_end])) 

    return damp


def peak_analysis(time, movement, joint, time_start, time1, time2): # window is in index length not seconds
    
    cross_damp = None
    
    #: Preprocess
    pre_peaks, _ = find_peaks(movement)
    if joint =='elbow' or joint == 'hand':
        #: Shoulder Pre processing of peaks
        min_peak = np.min(movement[pre_peaks])
        max_peak = np.max(movement[pre_peaks])
        min_i = np.where(movement == min_peak)[0]
        max_i = np.where(movement == max_peak)[0]
        min_peak_width = peak_widths(movement, min_i, rel_height=1)[0][0] #largeur du pic en bas pour le pic avec la plus petite hauteur
        max_peak_width = peak_widths(movement, max_i, rel_height=1)[0][0] #largeur du pic en haut pour le pic avec la plus grande hauteur
        peaks, _ = find_peaks(movement, height=(max_peak-min_peak)*0.035, width=[min_peak_width/100, max_peak_width])
    elif joint =='shoulder':
        # No preprocess for elbow so far
        peaks = pre_peaks
        min_peak = np.min(movement[pre_peaks])
        peaks, _ = find_peaks(movement, threshold=min_peak/1000) # indices de tous les peaks
    
    cross_peaks = peaks
    
    # Eliminate start sequence
    ts = np.where(time == time_start)[0]
    cross_peaks = cross_peaks[cross_peaks >= ts]
    
    #Number of peaks
    n_cross_peaks = len(cross_peaks)                    # number of peaks due to movement
    n_peaks_first = len(cross_peaks[(cross_peaks >= time1) & (cross_peaks < time2)])
    n_peaks_second = len(cross_peaks[cross_peaks >= time2])

    # Damping
    s_cross = cross_peaks[np.argsort(movement[cross_peaks])][len(cross_peaks)-2:len(cross_peaks)] # For Step input extract the 2 highest peaks : Adapt if other input
    l = len(s_cross)
    _cross_damp = np.zeros(l)
    s_cross = np.sort(s_cross) #indices des 2 pics les plus grands dans l'ordre croissant

    #Compute damping for the 2 perturbations or for the two parts of the movement
    if l > 1:
        c_peaks = np.asarray([x for x in cross_peaks if (x < s_cross[1] and x >= s_cross[0])]) 
        w = int(peak_widths(movement, [s_cross[0]])[0][0]) # window to refine damping  
        _cross_damp[0] = damping(movement, time, c_peaks, w, s_cross[0])
        c_peaks = np.asarray([x for x in cross_peaks if x >= s_cross[1]])
        w = int(peak_widths(movement, [s_cross[1]])[0][0]) # window to refine damping  
        _cross_damp[1] = damping(movement, time, c_peaks, w, s_cross[1])

        cross_damp_first = _cross_damp[0]
        cross_damp_second = _cross_damp[1]
    else:
        cross_damp_first = 0
        cross_damp_second = 0

    cross_damp = np.mean(_cross_damp)  # moyenne du damping pour les 2 periodes du mouvement

    return cross_peaks, n_cross_peaks, n_peaks_first, n_peaks_second, cross_damp, cross_damp_first, cross_damp_second
    

def speed_arc_length(movement):
    norm_mov = movement/max(movement)
    # Calculate arc length
    speed_arc_length = -sum(np.sqrt(pow(1/len(movement), 2) + pow(np.diff(norm_mov), 2)))
    return speed_arc_length


def sal(movement, fs, padlevel=4, fc=20.0):

    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    fc_inx = ((f <= fc) * 1).nonzero()[0]
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Calculate arc length
    sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return sal


def sparc(movement, fs, padlevel=4, fc=20.0, amp_th=0.05):
    """
    Calculates the smoothness of the given speed profile using the modified
    spectral arc length metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]
    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magnitude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.
    Notes
    -----
    This is the modified spectral arc length metric, which has been tested only
    for discrete movements.
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = sparc(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'
    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()[0]
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    if len(inx) > 0:
        fc_inx = range(0, min(inx[-1] + 1, len(f_sel)))  #inx[0]
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)

def dimensionless_jerk(movement, fs, data_type='speed'):
    """
    Calculates the smoothness metric for the given movement data using the
    dimensionless jerk metric. The input movement data can be 'speed',
    'acceleration' or 'jerk'.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}
    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'
    """
    # first ensure the movement type is valid.
    if data_type in ('speed', 'accl', 'jerk'):
        # first enforce data into an numpy array.
        movement = np.array(movement)

        # calculate the scale factor and jerk.
        movement_peak = max(abs(movement))
        dt = 1. / fs
        movement_dur = len(movement) * dt
        # get scaling factor:
        _p = {'speed': 3,                                           #ASK pourquoi 3 et pas 5 ?
              'accl': 1,
              'jerk': -1}
        p = _p[data_type]
        scale = pow(movement_dur, p) / pow(movement_peak, 2)

        # estimate jerk
        if data_type == 'speed':
            jerk = np.diff(movement, 2) / pow(dt, 2)
        elif data_type == 'accl':
            jerk = np.diff(movement, 1) / pow(dt, 1)
        else:
            jerk = movement

        # estimate dj
        return - scale * sum(pow(jerk, 2)) * dt
    else:
        raise ValueError('\n'.join(("The argument data_type must be either",
                                    "'speed', 'accl' or 'jerk'.")))

def log_dimensionless_jerk(movement, fs, data_type='speed'):
    """
    Calculates the smoothness metric for the given movement data using the
    log dimensionless jerk metric. The input movement data can be 'speed',
    'acceleration' or 'jerk'.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}
    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> ldl = log_dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'
    """
    return -np.log(abs(dimensionless_jerk(movement, fs, data_type)))