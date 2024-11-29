""" Compute metrics from osim simulation data """

from queue import Empty
from shutil import move
import sys
from tkinter import DoubleVar
from tokenize import Double
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.signal import *
from metrics import *

def main():   

    # set scenario
    traj = "flexion"  # "flexion" or "circle"
    pairs = True  # true to analyse pair scenaruio
    sim_perturb = False  # True to analyse perturbation scenario
    
    if not pairs:
        fixed_pathways= [''] 
        groups = ['Ia_Mn', 'Ia_In', 'Ia_Mn_syn', 'Ib','II', 'Rn_Mn']  
        
    # traj target
    if traj == "flexion":
        target_traj = "flexion_slow_2"
        period = 2.8  # in sec
    elif traj == "circle":
        target_traj = "circles_mod_4"
        period = 1.3  # in sec
    target_folder = "data/UP002/" + target_traj.split("_")[0] + "/" + target_traj.split("_")[1]+ "/" +  target_traj.split("_")[2] + "/" 
    target_file = target_folder + target_traj.split("_")[0]+"_" + target_traj.split("_")[1]+"_"+target_traj.split("_")[2]+"_position.txt"
    
    # Perturbation time
    perturb_time = 2*period/5

    # Metrics in ['rmse', 'peaks', 'sparc', 'sal', 'log_jerk', 'range_pos', 'damping', 'speed_arc_length', 'EMG']
    joint_list     = ['elbow']
    param_list     = ['kinematics_pos', 'kinematics_vel', 'speed']                                              
    metric_list    = ['rmse', 'sal', 'sparc', 'log_jerk', 'range_pos', 'speed_arc_length'] #, 'EMG'] 
    w_neg_emg = 0.5
    perturb_list    = ['rmse', 'deviation', 'sal', 'sparc', 'log_jerk', 'peaks', 'damping', 'speed_arc_length'] 

    if sim_perturb:
        main_folder = "results/" + traj + "/perturbation/control_input_nosc/"
    else:
        main_folder = "results/" + traj + "/movement/control_input_nosc/" 

    #:Osim settings
    step_size = 0.01
    delay = 0  # delay before controls in sec

    # Take params from                  : ['hand_kinematics_pos', 'hand_kinematics_vel', 'hand_speed']    
    # Take metrics from                 : ['rmse', 'peaks', 'sparc', 'sal', 'jerk', 'log_jerk', 'range_pos', 
    #                                       'damping', 'speed_arc_length', 'EMG']
    # Take joints from                  : ['hand', 'elbow', 'shoulder']                     
    # Take groups from                  : ['Ia_Mn', 'Ia_Mn_syn', 'Ia_In','Ib','II', 'Rn_Mn'] 
    weights = np.arange(0, 1.1, 0.1)

    #: Perturbations
    perturb = None 
    if sim_perturb:
        perturb = {'body': 'r_hand', 'force': [30], 'dir': [1], 'onset': [perturb_time], 'delta_t': 30} 
        metric_list = perturb_list
    
    # Compute metrics
    if not pairs:
        for pathway in fixed_pathways: 
            current_folder = main_folder+ 'fixed_'+pathway+"/"

            all_model = [f"{group}_{w}" for group in groups for w in weights]

            for joint in joint_list:

                # Joint_target['TRIlong', 'TRIlat']
                joints_target = pd.read_csv(target_file, sep=' ')
                if joint == "shoulder":
                    joint_target = joints_target.iloc[:, 0].values
                if joint == "elbow":
                    joint_target = joints_target.iloc[:, 1].values

                cp_metrics_all = []
                all_groups = []
                for group in groups:
                    cp_metrics = []
                    current_save,model_list = select_model_list(group, current_folder, weights)
                    for model in model_list:
                        save_folder = current_save + model + '/'
                        #: Sim Data
                        params = get_params(param_list, save_folder)
                        time = params[0].get('time').to_numpy()
                        #print("time", len(time))
                        #: Metrics
                        show_plot = False
                        [_cp_metrics , columns] = compute_metrics(model, joint, metric_list, params, save_folder, time, step_size, perturb_time, delay, perturb, w_neg_emg, show_plot, joint_target)
                        cp_metrics.append(_cp_metrics)
                        if all_model is not None:
                            cp_metrics_all.append(_cp_metrics)
                    metrics = pd.DataFrame(cp_metrics, columns = columns, index = model_list)
                    metrics['group'] = group
                    all_groups.extend(list(metrics['group'].values))
                    metrics.to_excel(current_save + group + '_' +  joint + '_' + 'metrics2.xlsx')

        if all_model is not None:
            metrics_all = pd.DataFrame(cp_metrics_all, columns = columns, index = all_model)
            metrics_all['group'] = all_groups
            metrics_all.to_excel(current_folder + joint + '_' + 'metric2s.xlsx')
        
    else:
        pairs = ['Ia_Mn','Ia_In']
        current_save = main_folder + 'pair_' + pairs[0]+ '_' +  pairs[1] + '/'
        model_list1 = [pairs[0] + '_' +  str(int(w*10)) for w in weights]
        model_list2 = [pairs[1] + '_' +  str(int(w*10)) for w in weights]
        
        for joint in joint_list:

            # Joint_target
            joints_target = pd.read_csv(target_file, sep=' ')
            if joint == "shoulder":
                joint_target = joints_target.iloc[:, 0].values
            if joint == "elbow":
                joint_target = joints_target.iloc[:, 1].values

            cp_metrics_all = []
            group2_column = []
            group1_column = []

            for model1 in model_list1:
                cp_metrics = []
                save_model1 = save_folder = current_save + model1 + '/' 
                for model2 in model_list2:
                    save_folder = current_save + model1 + '/' + model2 + '/'
                    #: Sim Data
                    params = get_params(param_list, save_folder)
                    time = params[0].get('time').to_numpy()
                    #print("time", len(time))
                    #: Metrics
                    [_cp_metrics , columns] = compute_metrics(model1, joint, metric_list, params, save_folder, time, step_size, perturb_time, delay, perturb, w_neg_emg, show_plot=False, joint_target=joint_target, model2=model2)
                    cp_metrics.append(_cp_metrics)
                    cp_metrics_all.append(_cp_metrics)
                metrics = pd.DataFrame(cp_metrics, columns = columns)
                metrics["pair1"] = model1
                metrics['pair2'] = model_list2
                group1_column.extend([model1]*len(model_list2))
                group2_column.extend(model_list2)
                metrics.to_excel(save_model1 + model1 + '_' +  joint + '_' + 'metrics2.xlsx')
            
        metrics_all = pd.DataFrame(cp_metrics_all, columns = columns)
        metrics_all['pair1'] = group1_column
        metrics_all['pair2'] = group2_column      
        metrics_all.to_excel(current_save + joint + '_' + 'metric2s.xlsx')
                    

def get_params(param_list, save_folder):
    "extract metrics from a osim result file"
    params = []
    files = {
            "kinematics_pos": "body_simulation_BodyKinematics_pos_global.sto",
            "kinematics_vel": "body_simulation_BodyKinematics_vel_global.sto",
            "speed": "simulation_States.sto",
            "pos": "simulation_States.sto",
            }
    for param in param_list:
        if param in files:
            file = open(save_folder + files[param], "r")
        else:
            sys.exit("Could not find metric, please check list")
        lines = file.readlines()
        #: Restructure parameters in a tab
        endhead_pos = 1
        while (lines[endhead_pos][0:9] != 'endheader'):
            endhead_pos += 1  
        len_col = len(lines[endhead_pos+1].split("	"))
        len_lines = len(lines) - (endhead_pos+2)
        param_temp = np.zeros((len_lines, len_col))
        columns = lines[endhead_pos+1].rstrip('\n').split("	")
        
        for i in range(endhead_pos+2, len(lines)):                                        
            param_temp[i-(endhead_pos+2),:] = list(map(float, (lines[i].rstrip('\n').split("	"))))   
        param_pd = pd.DataFrame(param_temp, columns = columns)
        params.append(param_pd) 
        
    return params


def compute_metrics(model, joint, metric_list, params, save_folder, time, step_size, perturb_time, delay, perturb, w_neg, show_plot=False, joint_target=None, model2=None):
    " Draft of function to compute metrics and compare them along different models"
    #: Sampling frequency
    fs = 1./step_size
    #: Analysis settings

    if params is not None:
        columns = []
        list = []
        #: Hand
        if 'hand' == joint:
            speed = np.sqrt(np.power(params[1].get('r_hand_X').to_numpy(), 2) + np.power(params[1].get('r_hand_Y').to_numpy(), 2))
            pos   = np.sqrt(np.power(params[0].get('r_hand_X').to_numpy(), 2) + np.power(params[0].get('r_hand_Y').to_numpy(), 2)) 
        #: Elbow
        if 'elbow' == joint:
            speed = params[2].get('/jointset/elbow/r_elbow_flexion/speed').to_numpy()
            pos   = params[2].get('/jointset/elbow/r_elbow_flexion/value').to_numpy()
        #: Shoulder 
        if 'shoulder' == joint:
            speed = params[2].get('/jointset/shoulder/r_shoulder_elev/speed').to_numpy()
            pos   = params[2].get('/jointset/shoulder/r_shoulder_elev/value').to_numpy()
        marker_v = speed 

        if perturb is not None:
            time1 = np.where(time>perturb["onset"][0])[0][0] #time first perturbation
            if len(perturb["onset"]) > 1:
                time2 = np.where(time>perturb["onset"][1])[0][0] #time second perturbation
                marker_v_second = marker_v[time2:]
            else:
                time2 = len(time)-1
            marker_v_first = marker_v[time1:time2-1]
        else:
            time1 = 0
            time2 = len(time)-1

        #: Peak Analysis
        cross_peaks, n_cross_peaks, n_peaks_first, n_peaks_second, cross_damp, cross_damp_first, cross_damp_second = peak_analysis(time, marker_v, joint, delay, time1, time2)

        #: PLOTS
        #plot_metrics(joint, params, model, save_folder, marker_v, time, cross_peaks, show_plot, time1, time2)  ###

        muscle_list_flex =['DELT1', 'BIClong', 'BICshort', 'BRA']
        muscle_list_ext = ['TRIlong', 'TRIlat']
        if 'rmse' in metric_list:
            columns.append('rmse')
            _rmse = rmse(pos, joint_target)
            list.append(_rmse)
        if 'EMG' in metric_list:
            columns.append('EMG_flex')
            columns.append('EMG_ext')
            columns.append('EMG_mean')
            _emg_flex, _emg_ext = comp_emg(save_folder, muscle_list_flex, muscle_list_ext, w_neg)
            list.append(_emg_flex)
            list.append(_emg_ext)
            list.append((_emg_flex+_emg_ext)/2)
        if 'EMG_mean' in metric_list:
            columns.append('EMG_mean')
            _emg_flex, _emg_ext = comp_emg(save_folder, muscle_list_flex, muscle_list_ext, w_neg)
            list.append((_emg_flex+_emg_ext)/2)
        if 'deviation' in metric_list:
            columns.append('deviation')
            _dev = deviation(pos, perturb_time, step_size)
            list.append(_dev)
        if 'peaks' in metric_list:
            columns.append('peaks')
            list.append(n_cross_peaks)
            columns.append('peaks_first')
            list.append(n_peaks_first)
            if len(perturb["onset"]) > 1:
                columns.append('peaks_second')
                list.append(n_peaks_second)
        if 'damping' in metric_list:
            columns.append('damping')
            list.append(cross_damp)
            columns.append('damping_first')
            list.append(cross_damp_first)
            if len(perturb["onset"]) > 1:
                columns.append('damping_second')
                list.append(cross_damp_second)
        if 'sparc':
            columns.append('sparc')
            _sparc, _, _= sparc(marker_v[time1:-1], fs)
            list.append(_sparc)
            if perturb is not None:
                columns.append('sparc_first')
                _sparc, _, _= sparc(marker_v_first,fs)
                list.append(_sparc)
                if len(perturb["onset"]) > 1:
                    columns.append('sparc_second')
                    _sparc, _, _= sparc(marker_v_second,fs)
                    list.append(_sparc)
        if 'sal' in metric_list:
            columns.append('sal')
            _sal = sal(marker_v[time1:-1],fs)
            list.append(_sal)
            if perturb is not None:
                columns.append('sal_first')
                _sal = sal(marker_v_first,fs)
                list.append(_sal)
                if len(perturb["onset"]) > 1:
                    columns.append('sal_second')
                    _sal = sal(marker_v_second,fs)
                    list.append(_sal)
        if 'speed_arc_length' in metric_list:
            columns.append('speed_arc_length')
            _speed_arc_length = speed_arc_length(marker_v[time1:-1])
            list.append(_speed_arc_length)
            if perturb is not None:
                columns.append('speed_arc_length_first')
                _speed_arc_length = speed_arc_length(marker_v_first)
                list.append(_speed_arc_length)
                if len(perturb["onset"]) > 1:
                    columns.append('speed_arc_length_second')
                    _speed_arc_length = speed_arc_length(marker_v_second)
                    list.append(_speed_arc_length)
        if 'jerk' in metric_list:
            columns.append('jerk')
            _jerk = dimensionless_jerk(marker_v[time1:-1], fs)
            list.append(_jerk) 
        if 'log_jerk' in metric_list:
            columns.append('log_jerk')
            _log_jerk = log_dimensionless_jerk(marker_v[time1:-1], fs)
            list.append(_log_jerk)
        if 'range_pos' in metric_list:
            columns.append('range_pos')
            _range = range_motion(pos)
            list.append(_range)
            if perturb is not None:
                columns.append('range_pos_first')
                _range = range_motion(pos[time1:time2-1])
                list.append(_range)
                if len(perturb["onset"]) > 1:
                    columns.append('range_pos_second')
                    _range = range_motion(pos[time2:])
                    list.append(_range)
        if 'range_speed' in metric_list:
            columns.append('range_speed')
            _range = range_motion(marker_v)
            list.append(_range)
            if perturb is not None:
                columns.append('range_speed_first')
                _range = range_motion(marker_v_first)
                list.append(_range)
                if len(perturb["onset"]) > 1:
                    columns.append('range_speed_second')
                    _range = range_motion(marker_v_second)
                    list.append(_range)

    return list, columns


def plot_metrics(joint, params, model, save_folder, marker_v, time, cross_peaks, show_plot, time1, time2):
    
    plt.figure(model + '_' + joint)
    if params is not None:
        plt.plot(time, marker_v, 'r', label="vel")
        plt.legend(loc='upper left')
        plt.xlabel("time [s]")
        plt.ylabel("vel [m/s]")
        plt.title(joint + '_' + 'peaks')
        plt.plot(time[cross_peaks], marker_v[cross_peaks], "x")
        plt.axvline(x=time[time1])
        plt.axvline(x=time[time2])
        #plt.ylim((0,15))
        plt.savefig(save_folder + joint + '_' + "metrics_plot")
            
        if show_plot:
            plt.show()
        
        plt.close('all')

def select_model_list(group, current_folder, weights):
    current_save = current_folder+group+'/'
    model_list = [group + '_' +  str(int(w*10)) for w in weights]
    return current_save,model_list

        
if __name__ == '__main__':
    main()