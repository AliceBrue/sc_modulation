""" functions to compute EMG activation time windows """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import opensim_sto_reader as os_read
from net_osim import *
from osim_model import *
import matplotlib.pylab as pylab


# plot params
params = {'legend.fontsize': 20,
         'axes.labelsize': 20,
         'axes.titlesize':24,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
pylab.rcParams.update(params)


def main():        
 
    # set scenario
    traj = "flexion"  # "flexion" or "circle"
    condition = 'movement'  # 'movement' or 'optimisation_bi' or 'optimisation_w'
    baseline = 1  
    optim_algo = "cma"
    with_sc = False  # True for the minimal SC scenario in condition='optimisation_bi'
    pairs = False
    # EMG comparison thresholds
    activation_th_flex=0.85
    activation_th_ext=0.65
    activation_len_t=100
    activation_len_s=100
    show_plot = False

    # target file
    if traj == "flexion":
        target_traj = "flexion_slow_2"
    elif traj == "circle":
        target_traj = "circles_mod_4"
    target_folder = "data/UP002/" + target_traj.split("_")[0] + "/" + target_traj.split("_")[1]+ "/" +  target_traj.split("_")[2] + "/" 
    emg_target_file = target_folder + "emg.txt"

    # folder
    if condition == "optimisation_bi" and with_sc:
        input_folder = "brain_input_SC"
    elif condition == "optimisation_bi":
        input_folder = "brain_input"
    elif condition == "optimisation_w":
        input_folder = "SC_weights"
    else:
        input_folder = "control_input_nosc/"

    subfolder_fixed = ['fixed_/']
    subfolder_pathways = ['Ia_Mn', 'Ia_In', 'Ia_Mn_syn', 'Ib','II', 'Rn_Mn']
    if pairs:
        pairs = ['Ia_Mn','Ia_In']

    muscle_list = ['DELT1', 'TRIlong', 'TRIlat', 'BIClong', 'BICshort', 'BRA']
    muscle_list_flex =['DELT1', 'BIClong', 'BICshort', 'BRA']
    muscle_list_ext = ['TRIlong', 'TRIlat']

    compute_activation_time_windows(condition, target_traj, pairs, baseline, input_folder, optim_algo, emg_target_file, muscle_list, muscle_list_flex, muscle_list_ext, subfolder_fixed, subfolder_pathways, activation_len_t, activation_len_s,  activation_th_flex, activation_th_ext, show_plot = show_plot)

    if show_plot:
        plt.show()


def compute_activation_time_windows(condition, target_traj, pairs, baseline, input_folder, optim_algo, emg_target_file, muscle_list, muscle_list_flex, muscle_list_ext, subfolder_fixed, subfolder_pathways, activation_len_t, activation_len_s, activation_th_flex, activation_th_ext, show_plot=True):
    
    if 'optimisation' in condition: 

        save_plot_path = "results/" + target_traj.split("_")[0] + "/optimisation/" + input_folder + "/" + optim_algo + "/baseline_" + str(baseline) + "_" + target_traj + "/"
        emg_sim = save_plot_path + "simulation_States.sto"
        activation_w_flex_file = save_plot_path + 'activation_flex_dict.txt'
        activation_w_ext_file = save_plot_path + 'activation_ext_dict.txt'
    
        time_t, emg_signals_t_flex, time_s, emg_signals_s_flex, emg_signals_t_ext, emg_signals_s_ext = extract_emg_wind(emg_sim, emg_target_file, muscle_list)

        overlap_flex, noverlap_flex = plot_emg_with_windows(save_plot_path, time_t, emg_signals_t_flex, time_s, emg_signals_s_flex, muscle_list_flex, 
                                                                    activation_len_t=activation_len_t, activation_len_s=activation_len_s, activation_th_flex=activation_th_flex, activation_th_ext=activation_th_ext)
        overlap_ext, noverlap_ext = plot_emg_with_windows(save_plot_path, time_t, emg_signals_t_ext, time_s, emg_signals_s_ext, muscle_list_ext, 
                                                                                activation_len_t=activation_len_t, activation_len_s=activation_len_s, activation_th_flex=activation_th_flex, activation_th_ext=activation_th_ext)
        
        write_dict_to_file(overlap_flex, "_overlap", activation_w_flex_file, 'w')
        write_dict_to_file(noverlap_flex, "_noverlap", activation_w_flex_file, 'a')
        write_dict_to_file(overlap_ext, "_overlap", activation_w_ext_file, 'w')
        write_dict_to_file(noverlap_ext, "_noverlap", activation_w_ext_file, 'a')


    elif condition=='movement' and not pairs:

        # Set the fixed base directory
        base_directory = "results/"  + target_traj.split("_")[0] + "/" + condition + "/" + input_folder
        weights=np.arange(0, 1.1, 0.1)        

        for dir in subfolder_fixed:
            current_dir=base_directory+dir
            for pathway in subfolder_pathways:
                current_dir_p=current_dir+pathway+'/'
                for w in weights:
                    current_dir_w=current_dir_p+pathway+'_'+str(int(w*10))+'/'
                    emg_sim=current_dir_w+'simulation_States.sto'
                    activation_w_flex_file=current_dir_w+'activation_flex_dict.txt'
                    activation_w_ext_file=current_dir_w+'activation_ext_dict.txt'
                    save_plot_path=current_dir_w
                    
                    time_t, emg_signals_t_flex, time_s, emg_signals_s_flex, emg_signals_t_ext, emg_signals_s_ext = extract_emg_wind(emg_sim, emg_target_file, muscle_list)

                    overlap_flex, noverlap_flex = plot_emg_with_windows(save_plot_path, time_t, emg_signals_t_flex, time_s, emg_signals_s_flex, muscle_list_flex, 
                                                                                activation_len_t=activation_len_t, activation_len_s=activation_len_s, activation_th_flex=activation_th_flex, activation_th_ext=activation_th_ext)
                    overlap_ext, noverlap_ext = plot_emg_with_windows(save_plot_path, time_t, emg_signals_t_ext, time_s, emg_signals_s_ext, muscle_list_ext, 
                                                                                            activation_len_t=activation_len_t, activation_len_s=activation_len_s, activation_th_flex=activation_th_flex, activation_th_ext=activation_th_ext)
                    
                    write_dict_to_file(overlap_flex, "_overlap", activation_w_flex_file, 'w')
                    write_dict_to_file(noverlap_flex, "_noverlap", activation_w_flex_file, 'a')
                    write_dict_to_file(overlap_ext, "_overlap", activation_w_ext_file, 'w')
                    write_dict_to_file(noverlap_ext, "_noverlap", activation_w_ext_file, 'a')
                        
    
    elif condition=='movement' and pairs:
        pairs = ['Ia_Mn','Ia_In']

        # Set the fixed base directory
        base_directory = "results/" + target_traj.split("_")[0] + "/" + condition + "/" + input_folder + 'pair_' + pairs[0]+ '_' +  pairs[1] + '/'
        weights=np.arange(0, 1.1, 0.1)
        model_list1 = [pairs[0] + '_' +  str(int(w*10)) for w in weights]
        model_list2 = [pairs[1] + '_' +  str(int(w*10)) for w in weights]   

        for dir in model_list1:
            for dir2 in model_list2:
                current_dir=base_directory+dir+"/"+dir2+"/"
                emg_sim=current_dir+'simulation_States.sto'
                activation_w_flex_file=current_dir+'activation_flex_dict.txt'
                activation_w_ext_file=current_dir+'activation_ext_dict.txt'
                save_plot_path=current_dir

                time_t, emg_signals_t_flex, time_s, emg_signals_s_flex, emg_signals_t_ext, emg_signals_s_ext = extract_emg_wind(emg_sim, emg_target_file, muscle_list)

                overlap_flex, noverlap_flex = plot_emg_with_windows(save_plot_path, time_t, emg_signals_t_flex, time_s, emg_signals_s_flex, muscle_list_flex, 
                                                                            activation_len_t=activation_len_t, activation_len_s=activation_len_s, activation_th_flex=activation_th_flex, activation_th_ext=activation_th_ext)
                overlap_ext, noverlap_ext = plot_emg_with_windows(save_plot_path, time_t, emg_signals_t_ext, time_s, emg_signals_s_ext, muscle_list_ext, 
                                                                                        activation_len_t=activation_len_t, activation_len_s=activation_len_s, activation_th_flex=activation_th_flex, activation_th_ext=activation_th_ext)
                
                write_dict_to_file(overlap_flex, "_overlap", activation_w_flex_file, 'w')
                write_dict_to_file(noverlap_flex, "_noverlap", activation_w_flex_file, 'a')
                write_dict_to_file(overlap_ext, "_overlap", activation_w_ext_file, 'w')
                write_dict_to_file(noverlap_ext, "_noverlap", activation_w_ext_file, 'a')

    if show_plot:
        plt.show()


def extract_emg_wind(emg_sim, emg_target_file, muscle_list):
    
    emg=read_data(emg_sim)

    f_t=open(emg_target_file,"r")
    lines_t=f_t.readlines()[1:]
    time_t=[]
    emg_DELT1=[]
    emg_TRIlong=[]
    emg_TRIlat=[]
    emg_BIClong=[]
    emg_BICshort=[]
    emg_BRA=[]

    for x_t in lines_t:
        time_t.append(x_t.split(',')[0])
        emg_DELT1.append(x_t.split(',')[1])
        emg_TRIlong.append(x_t.split(',')[3])
        emg_TRIlat.append(x_t.split(',')[4])
        emg_BIClong.append(x_t.split(',')[5])
        emg_BICshort.append(x_t.split(',')[6])
        emg_BRA.append(x_t.split(',')[7])
        f_t.close()

    time_t = np.array(time_t).astype(float)

    emg_DELT1=np.array(emg_DELT1)
    emg_DELT1=emg_DELT1.astype(np.float32)

    emg_TRIlong=np.array(emg_TRIlong)
    emg_TRIlong=emg_TRIlong.astype(np.float32)

    emg_TRIlat=np.array(emg_TRIlat)
    emg_TRIlat=emg_TRIlat.astype(np.float32)

    emg_BIClong=np.array(emg_BIClong)
    emg_BIClong=emg_BIClong.astype(np.float32)

    emg_BICshort=np.array(emg_BICshort)
    emg_BICshort=emg_BICshort.astype(np.float32)

    emg_BRA=np.array(emg_BRA)
    emg_BRA=emg_BRA.astype(np.float32)

    # Interp target EMG
    emg_target = emg.copy()
    time_s = emg["time"]
    emg_target["time"] = np.interp(time_s, time_t, time_t)
    emg_target["DELT1"] = np.interp(time_s, time_t, emg_DELT1)
    emg_target["TRIlong"] = np.interp(time_s, time_t, emg_TRIlong)
    emg_target["TRIlat"] = np.interp(time_s, time_t, emg_TRIlat)
    emg_target["BIClong"] = np.interp(time_s, time_t, emg_BIClong)
    emg_target["BICshort"] = np.interp(time_s, time_t, emg_BICshort)
    emg_target["BRA"] = np.interp(time_s, time_t, emg_BRA)

    rms_val=dict()
    for i in range(len(muscle_list)):
        if muscle_list[i]=='DELT1':
            rms_val['DELT1_t']=rms(emg_DELT1)
            rms_val['DELT1_s']=rms(emg['DELT1']) 
        elif muscle_list[i]=='TRIlong':
            rms_val['TRIlong_t']=rms(emg_TRIlong)
            rms_val['TRIlong_s']=rms(emg['TRIlong'])
        elif muscle_list[i]=='TRIlat':
            rms_val['TRIlat_t']=rms(emg_TRIlat)
            rms_val['TRIlat_s']=rms(emg['TRIlat'])
        elif muscle_list[i]=='BIClong':
            rms_val['BIClong_t']=rms(emg_BIClong)
            rms_val['BIClong_s']=rms(emg['BIClong'])
        elif muscle_list[i]=='BICshort':
            rms_val['BICshort_t']=rms(emg_BICshort)
            rms_val['BICshort_s']=rms(emg['BICshort'])
        elif muscle_list[i]=='BRA':
            rms_val['BRA_t']=rms(emg_BRA)
            rms_val['BRA_s']=rms(emg['BRA'])
    
    #write_dict_to_file(rms_val, rms_file, 'w')

    time_t=emg_target['time']
    emg_signals_t_flex = [emg_target['DELT1'], emg_target['BIClong'], emg_target['BICshort'], emg_target['BRA']]
    emg_signals_t_ext = [emg_target['TRIlong'], emg_target['TRIlat']]
    emg_signals_s_flex = [emg['DELT1'], emg['BIClong'], emg['BICshort'], emg['BRA']]
    emg_signals_s_ext = [emg['TRIlong'], emg['TRIlat']]

    return time_t, emg_signals_t_flex, time_s, emg_signals_s_flex, emg_signals_t_ext, emg_signals_s_ext
    

def read_data(simulation_file):

    OS=os_read.readMotionFile(simulation_file)
    labels=OS[1]
    values=OS[2]
    n_lines=len(values)
    n_col=len(labels)
    emg={}
    emg['time']=[0]*n_lines
    emg['DELT1']=[0]*n_lines
    emg['TRIlong']=[0]*n_lines
    emg['TRIlat']=[0]*n_lines
    emg['TRImed']=[0]*n_lines
    emg['BIClong']=[0]*n_lines
    emg['BICshort']=[0]*n_lines
    emg['BRA']=[0]*n_lines

    for i in range (n_col):
        if labels[i]=="/forceset/DELT1/activation":
            for j in range(n_lines):
                data_delt1=values[j]
                emg['DELT1'][j]=data_delt1[i]
                emg['time'][j]=data_delt1[0]

        if labels[i]=="/forceset/TRIlong/activation":
            for j in range(n_lines):
                data_trilong=values[j]
                emg['TRIlong'][j]=data_trilong[i]
        if labels[i]=="/forceset/TRIlat/activation":
            for j in range(n_lines):
                data_trilat=values[j]
                emg['TRIlat'][j]=data_trilat[i]
        if labels[i]=="/forceset/TRImed/activation":
            for j in range(n_lines):
                data_trimed=values[j]
                emg['TRImed'][j]=data_trimed[i]
        if labels[i]=="/forceset/BIClong/activation":
            for j in range(n_lines):
                data_bcilong=values[j]
                emg['BIClong'][j]=data_bcilong[i]
        if labels[i]=="/forceset/BICshort/activation":
            for j in range(n_lines):
                data_bcishort=values[j]
                emg['BICshort'][j]=data_bcishort[i]
        if labels[i]=="/forceset/BRA/activation":
            for j in range(n_lines):
                data_bra=values[j]
                emg['BRA'][j]=data_bra[i]

    return emg


def rms(values):
    square = 0
    mean = 0
    rms= 0
    n=len(values)
    for i in range(0,n):
        square += (values[i]**2)

    mean = (square / (float)(n))
    rms = np.sqrt(mean)
    return rms


def write_dict_to_file(dict_data, sufix, file_name, mode='w'):
    with open(file_name, mode) as txt_file:
        for key, value in dict_data.items():
            txt_file.write(f"{key}{sufix}: {value}\n")


def count_consecutive(values):
    count = 0
    max_count = 0
    for value in values:
        if value is not None:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def remove_short_activations(time, window, lower_limit):

    consecutive_count = 0
    n=len(window)

    for i, value in enumerate(window):
        if value is not None:
            consecutive_count += 1
        else:
            if consecutive_count < lower_limit:
                for j in range(i - consecutive_count, i):
                    window[j] = None
                    time[j]=None
            consecutive_count = 0

    # Check if there are short activations at the end
    if consecutive_count < lower_limit:
        for j in range(n - consecutive_count, n):
            window[j] = None
            time[j] = None

    return window, time


def scatter_to_window(activation_windows, muscle_name):

    activation_window_times = {}
    activation_window_times["ti"] = []
    activation_window_times["tf"] = []

    if activation_windows[muscle_name][0][0] is not None:
        activation_window_times["ti"].append(0)

    for i in range(len(activation_windows[muscle_name][0])-1):
        if activation_windows[muscle_name][0][i] is None and activation_windows[muscle_name][0][i+1] is not None:
            ti = (i+1)/len(activation_windows[muscle_name][0])
            activation_window_times["ti"].append(ti)
        elif activation_windows[muscle_name][0][i] is not None and activation_windows[muscle_name][0][i+1] is None:
            tf = (i)/len(activation_windows[muscle_name][0])
            activation_window_times["tf"].append(tf)
    
    if activation_windows[muscle_name][0][-1] is not None:
        activation_window_times["tf"].append(1)

    return activation_window_times


def Activation(emg_signal_t,emg_signal_s,window_val_t, window_val_s, time_t, time_s, activation_th):

    window_t=[None]*len(emg_signal_t)
    time_window_t=[None]*len(emg_signal_t)
    window_s=[None]*len(emg_signal_s)
    time_window_s=[None]*len(emg_signal_s)

    for i in range (len(emg_signal_t)):
        if rms(emg_signal_t)>0.05 and emg_signal_t[i]>(activation_th*rms(emg_signal_t)):
            window_t[i]=window_val_t
            time_window_t[i]=time_t[i]
    for j in range (len(emg_signal_s)):
        if rms(emg_signal_s)>0.05 and emg_signal_s[j]>(activation_th*rms(emg_signal_s)):
            window_s[j]=window_val_s
            time_window_s[j]=time_s[j]

    return window_t, time_window_t, window_s, time_window_s


def comp_overlap(activation_windows_t, activation_windows_s, d_val):

    for i in range(len(activation_windows_t[0])):
        if activation_windows_t[0][i] is None:
            activation_windows_t[0][i] = -1
    
    for i in range(len(activation_windows_s[0])):
        if activation_windows_s[0][i] is None:
            activation_windows_s[0][i] = -1

    if np.sum(np.array((activation_windows_t[0])) > 0) > 0:
        pos_overlap = np.sum(np.round(np.array(activation_windows_s[0])-np.array((activation_windows_t[0])) - d_val, 2) == 0)/np.sum(np.array((activation_windows_t[0])) > 0)*100
    else:
        pos_overlap = 0
    neg_overlap = np.sum(np.abs(np.array(activation_windows_s[0])-np.array((activation_windows_t[0]))) > d_val)/len(activation_windows_t[0])*100
    
    """for i in range(len(activation_windows_t[0])):
        if activation_windows_t[0][i] != -1:
            activation_windows_t[0][i] = 2
    
    for i in range(len(activation_windows_s[0])):
        if activation_windows_s[0][i] != -1:
            activation_windows_s[0][i] = 2
            
    acti_overlap = np.sum(np.array(activation_windows_s[0])-np.array((activation_windows_t[0])) != 0)/len(activation_windows_s[0])*100"""

    return pos_overlap, neg_overlap
    

def plot_emg_with_windows(save_path, time_t, emg_signal_t, time_s, emg_signal_s, muscle_list, activation_len_t=250, activation_len_s=100, activation_th_flex=0.85, activation_th_ext=0.65):

    activation_windows_t = {}
    activation_windows_s = {}

    activation_windows_tt = {}
    activation_windows_ts = {}

    overlap = {}
    noverlap = {}

    fig, ax = plt.subplots(figsize=(6,5))
    label=True
    d_val = 0.1  # difference between y value set to EMG and sim time windows
    for i in range(len(muscle_list)):
        muscle_name = muscle_list[i]
        window_val_t = 0
        if muscle_name == 'DELT1':
            window_val_t = 0.85
            window_val_s = window_val_t + d_val
            activation_th=activation_th_flex
        elif muscle_name == 'TRIlong':
            window_val_t= 0.7
            window_val_s= window_val_t + d_val
            activation_th=activation_th_ext
        elif muscle_name == 'TRIlat':
            window_val_t= 0.5
            window_val_s= window_val_t + d_val
            activation_th=activation_th_ext
        elif muscle_name == 'BIClong':
            window_val_t= 0.65
            window_val_s= window_val_t + d_val
            activation_th=activation_th_flex
        elif muscle_name == 'BICshort':
            window_val_t= 0.45
            window_val_s= window_val_t + d_val
            activation_th=activation_th_flex
        else:
            window_val_t= 0.25
            window_val_s= window_val_t + d_val
            activation_th=activation_th_flex

        window_t, time_window_t, window_s, time_window_s = Activation(emg_signal_t[i], emg_signal_s[i], window_val_t, window_val_s,time_t, time_s, activation_th)

        activation_windows_t[muscle_name] = remove_short_activations(time_window_t, window_t, activation_len_t)
        activation_windows_s[muscle_name] = remove_short_activations(time_window_s, window_s, activation_len_s)

        activation_windows_tt[muscle_name] = scatter_to_window(activation_windows_t, muscle_name)
        activation_windows_ts[muscle_name] = scatter_to_window(activation_windows_s, muscle_name)

        colors=['blue', 'green', 'tab:red', 'orange']
        if label:
            plt.plot(time_t, emg_signal_t[i], label=f'{muscle_list[i]} - Target EMG', linewidth=2, color=colors[i])
            plt.plot(time_s, emg_signal_s[i], label=f'{muscle_list[i]} - Simulated EMG', linewidth=2, linestyle='dashed',color=colors[i])
            label=False
        else:
            plt.plot(time_t, emg_signal_t[i], label=f'{muscle_list[i]}', linewidth=2, color=colors[i])
            plt.plot(time_s, emg_signal_s[i], linewidth=2, linestyle='dashed',color=colors[i])
        
        for w in range(len(activation_windows_tt[muscle_name]["ti"])):
            ti_t = activation_windows_tt[muscle_name]["ti"][w]
            tf_t = activation_windows_tt[muscle_name]["tf"][w]
            plt.axhline(window_val_t, ti_t, tf_t, color=colors[i])
        for w in range(len(activation_windows_ts[muscle_name]["ti"])):
            ti_s = activation_windows_ts[muscle_name]["ti"][w]
            tf_s = activation_windows_ts[muscle_name]["tf"][w]
            plt.axhline(window_val_s, ti_s, tf_s, linestyle="dashed", color=colors[i])
    
        # Compute metrics overlap
        overlap[muscle_name], noverlap[muscle_name] = comp_overlap(activation_windows_t[muscle_name], activation_windows_s[muscle_name], d_val)

    plt.xlabel('Time')
    plt.ylabel('EMG Signal')
    plt.ylim(0,1)
    plt.gca().xaxis.set_major_locator(MaxNLocator(prune='both'))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
    ax.spines[['right', 'top']].set_visible(False)
    plt.legend()
    
    if 'DELT1' in muscle_list:
        save_file_path = os.path.join(save_path, 'emg_flex.svg')
        plt.savefig(save_file_path)
    else:
        save_file_path = os.path.join(save_path, 'emg_ext.svg')
        plt.savefig(save_file_path)
    
    return overlap, noverlap
    

if __name__ == '__main__':
    main()
