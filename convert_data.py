""" functions to extract and align recording and simulation data """

from net_osim import *
from osim_model import *
from connections import *
from metrics import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def convertData_init_targ(file_time_emg, target_file):

    target=open(target_file,"r")
    lines_target=target.readlines()
    target_elbow_position=[]
    for x_target in lines_target:
        target_elbow_position.append(x_target.split(' ')[1])
    target.close()

    f_t=open(file_time_emg,"r")
    lines_t=f_t.readlines()[1:]
    time=[]
    for x_t in lines_t:
        time.append(x_t.split(',')[0])
    f_t.close()

    downsample_factor_targ=len(time)/len(target_elbow_position)
    length_new_time_targ= int(len(time) / downsample_factor_targ)
    time = [float(x) for x in time]
    time_new_targ=np.linspace(time[0], time[-1], length_new_time_targ)

    df = pd.DataFrame({'time_targ':time_new_targ, 'target_elbow_positions':target_elbow_position})
    df['time_targ']=df['time_targ'].astype(float)
    df['target_elbow_positions']=df['target_elbow_positions'].astype(float)

    df.plot('time_targ', 'target_elbow_positions', kind='line')


def convertData_pos_time_targ(ax, ax2, file_position, target_file, file_time_emg, scenario, plot_target=True, color=None):

    f_p=open(file_position,"r")
    lines_p=f_p.readlines()
    elbow_position=[]
    shoulder_position=[]
    elbow_vel=[]
    shoulder_vel=[]
    time_p=[]
    for x_p in lines_p:
        elbow_position.append(x_p.split(' ')[1])
        shoulder_position.append(x_p.split(' ')[2])
        elbow_vel.append(x_p.split(' ')[3])
        shoulder_vel.append(x_p.split(' ')[4])
        time_p.append(x_p.split(' ')[0])
    f_p.close()

    target=open(target_file,"r")
    lines_target=target.readlines()
    target_elbow_position=[]
    target_shoulder_position=[]
    for x_target in lines_target:
        target_elbow_position.append(x_target.split(' ')[1])
        target_shoulder_position.append(x_target.split(' ')[0])
    target.close()


    f_t=open(file_time_emg,"r")
    lines_t=f_t.readlines()[1:]
    time=[]
    for x_t in lines_t:
        time.append(x_t.split(',')[0])
    f_t.close()

    downsample_factor_targ=len(time)/len(target_elbow_position)
    length_new_time_targ= int(len(time) / downsample_factor_targ)
    time = [float(x) for x in time]
    time_new_targ=np.linspace(time[0], time[-1], length_new_time_targ)


    df_p = pd.DataFrame({'time_pos':time_p, 'elbow_positions':elbow_position, 'shoulder_positions':shoulder_position, 'shoulder_vel': shoulder_vel, 'elbow_vel':elbow_vel})
    df_t=pd.DataFrame({'time_targ':time_new_targ, 'target_elbow_positions':target_elbow_position, 'target_shoulder_positions':target_shoulder_position})
    df_p['time_pos']=df_p['time_pos'].astype(float)
    df_t['time_targ']=df_t['time_targ'].astype(float)
    df_p['elbow_positions']=df_p['elbow_positions'].astype(float)
    df_t['target_elbow_positions']=df_t['target_elbow_positions'].astype(float)
    df_p['shoulder_positions']=df_p['shoulder_positions'].astype(float)
    df_t['target_shoulder_positions']=df_t['target_shoulder_positions'].astype(float)
    df_p['shoulder_vel']=df_p['shoulder_vel'].astype(float)
    df_p['elbow_vel']=df_p['elbow_vel'].astype(float)
    
    # plot position
    if plot_target:
        ax[0].plot(df_t['time_targ'], df_t['target_shoulder_positions']*180/np.pi, "k", linewidth=2, label="target") #, linestyle="--")
        ax[1].plot(df_t['time_targ'], df_t['target_elbow_positions']*180/np.pi, "k", linewidth=2, label="target")
    if "optimisation" in scenario:
        if color is not None:
            ax[0].plot(df_p['time_pos'], df_p['shoulder_positions']*180/np.pi, color=color, label=scenario.split("optimisation_")[1])
            ax[1].plot(df_p['time_pos'], df_p['elbow_positions']*180/np.pi, color=color, label=scenario.split("optimisation_")[1])
        else:
            ax[0].plot(df_p['time_pos'], df_p['shoulder_positions']*180/np.pi, label=scenario.split("optimisation_")[1])
            ax[1].plot(df_p['time_pos'], df_p['elbow_positions']*180/np.pi, label=scenario.split("optimisation_")[1])
    else:
        if color is not None:
            ax[0].plot(df_p['time_pos'], df_p['shoulder_positions']*180/np.pi, color=color, label=scenario) #, linestyle="--")
            ax[1].plot(df_p['time_pos'], df_p['elbow_positions']*180/np.pi, color=color, label=scenario)
        else:
            ax[0].plot(df_p['time_pos'], df_p['shoulder_positions']*180/np.pi, label=scenario) #, linestyle="--")
            ax[1].plot(df_p['time_pos'], df_p['elbow_positions']*180/np.pi, label=scenario)
    ax[0].set_ylim(-20, 100)
    ax[1].set_ylim(0, 140)
    ax[0].set_xlabel('time(s)')
    ax[0].set_ylabel('Shoulder position (°)')
    ax[1].set_xlabel('time(s)')
    ax[1].set_ylabel('Elbow position (°)')
    ax[0].set_title('Shoulder position')
    ax[1].set_title('Elbow position')
    #ax[0].legend()
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[1].spines[['right', 'top']].set_visible(False)
    plt.tight_layout()

    # plot velocity
    if "optimisation" in scenario:
        if color is not None:
            ax2[0].plot(df_p['time_pos'], df_p['shoulder_vel']*180/np.pi, color=color, label=scenario.split("optimisation_")[1])
            ax2[1].plot(df_p['time_pos'], df_p['elbow_vel']*180/np.pi, color=color, label=scenario.split("optimisation_")[1])
        else:
            ax2[0].plot(df_p['time_pos'], df_p['shoulder_vel']*180/np.pi, label=scenario.split("optimisation_")[1])
            ax2[1].plot(df_p['time_pos'], df_p['elbow_vel']*180/np.pi, label=scenario.split("optimisation_")[1])
    else:
        if color is not None:
            ax2[0].plot(df_p['time_pos'], df_p['shoulder_vel']*180/np.pi, color=color, label=scenario)
            ax2[1].plot(df_p['time_pos'], df_p['elbow_vel']*180/np.pi, color=color, label=scenario)
        else:
            ax2[0].plot(df_p['time_pos'], df_p['shoulder_vel']*180/np.pi, label=scenario)
            ax2[1].plot(df_p['time_pos'], df_p['elbow_vel']*180/np.pi, label=scenario)
    ax2[0].set_ylim(-60, 60)
    ax2[1].set_ylim(-140, 140)
    ax2[0].set_xlabel('time(s)')
    ax2[0].set_ylabel('Shoulder velocity (°/s)')
    ax2[1].set_xlabel('time(s)')
    ax2[1].set_ylabel('Elbow velocity (°/s)')
    ax2[0].set_title('Shoulder velocity')
    ax2[1].set_title('Elbow velocity')
    #ax2[0].legend()
    ax2[0].spines[['right', 'top']].set_visible(False)
    ax2[1].spines[['right', 'top']].set_visible(False)
    plt.tight_layout()


def convertData_data(file_position):
    
    f_p=open(file_position,"r")
    lines_p=f_p.readlines()
    elbow_position=[]
    time_p=[]
    for x_p in lines_p:
        elbow_position.append(x_p.split(' ')[1])
        time_p.append(x_p.split(' ')[0])
    f_p.close()

    df_p = pd.DataFrame({'time_pos':time_p, 'elbow_positions':elbow_position})
    df_p['time_pos']=df_p['time_pos'].astype(float)
    df_p['elbow_positions']=df_p['elbow_positions'].astype(float)
    df_p.plot('time_pos', 'elbow_positions', kind='line')
    plt.xlabel('time(s)')
    plt.ylabel('Elbow position')


def plot_params(ax, ax2, params_file, color=None, label=None, label_fe=True):

    f_p=open(params_file,"r")
    lines_param=f_p.readlines()
    iterations=[]
    loss=[]
    offset_flex=[]
    offset_ext=[]
    frequency_flex=[]
    frequency_ext=[]
    amplitude_flex=[]
    amplitude_ext=[]
    phase_flex=[]
    phase_ext=[]
    for line_number, x_p in enumerate(lines_param, start=1):
    # Skip the first line
        if line_number==1:
            continue
        iteration=x_p.split(', ')[0]
        if  iteration not in iterations:
            iterations.append(iteration)
            loss.append(x_p.split(', ')[1])
            offset_flex.append(x_p.split(', ')[2])
            frequency_flex.append(x_p.split(', ')[3])
            amplitude_flex.append(x_p.split(', ')[4])
            phase_flex.append(x_p.split(', ')[5])
            offset_ext.append(x_p.split(', ')[6])
            frequency_ext.append(x_p.split(', ')[7])
            amplitude_ext.append(x_p.split(', ')[8])
            phase_ext.append(x_p.split(', ')[9])
        
    f_p.close()

    df_loss = pd.DataFrame({'Iteration':iterations, 'loss':loss})
    df_loss['Iteration']=df_loss['Iteration'].astype(float)
    df_loss['loss']=df_loss['loss'].astype(float)
    df_offset = pd.DataFrame({'Iteration':iterations, 'offset_flex':offset_flex, 'offset_ext':offset_ext})
    df_offset['Iteration']=df_offset['Iteration'].astype(float)
    df_offset['offset_flex']=df_offset['offset_flex'].astype(float)
    df_offset['offset_ext']=df_offset['offset_ext'].astype(float)
    df_freq = pd.DataFrame({'Iteration':iterations, 'frequency_flex':frequency_flex, 'frequency_ext':frequency_ext})
    df_freq['Iteration']=df_freq['Iteration'].astype(float)
    df_freq['frequency_flex']=df_freq['frequency_flex'].astype(float)
    df_freq['frequency_ext']=df_freq['frequency_ext'].astype(float)
    df_amp = pd.DataFrame({'Iteration':iterations, 'amplitude_flex':amplitude_flex, 'amplitude_ext':amplitude_ext})
    df_amp['Iteration']=df_amp['Iteration'].astype(float)
    df_amp['amplitude_flex']=df_amp['amplitude_flex'].astype(float)
    df_amp['amplitude_ext']=df_amp['amplitude_ext'].astype(float)
    df_phase = pd.DataFrame({'Iteration':iterations, 'phase_flex':phase_flex, 'phase_ext':phase_ext})
    df_phase['Iteration']=df_phase['Iteration'].astype(float)
    df_phase['phase_flex']=df_phase['phase_flex'].astype(float)
    df_phase['phase_ext']=df_phase['phase_ext'].astype(float)

    # figure loss
    if color is not None:
        if label is not None:
            ax.plot(df_loss['Iteration'], df_loss['loss'], color=color, label=label)
        else:
            ax.plot(df_loss['Iteration'], df_loss['loss'], color=color)
    else:
        if label is not None:
            ax.plot(df_loss['Iteration'], df_loss['loss'], label=label)
        else:
            ax.plot(df_loss['Iteration'], df_loss['loss'])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (rad)')
    ax.set_title('Loss')
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    

    # figure parameters
    if color is not None:
        if label_fe:
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_flex'], label="flex", color=color)
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_ext'], "--", label="ext", color=color)
        else:
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_flex'], color=color)
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_ext'], "--", color=color)
    else:
        if label_fe:
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_flex'], label="flex")
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_ext'], "--", label="ext")
        else:
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_flex'])
            ax2[0,0].plot(df_offset['Iteration'], df_offset['offset_ext'], "--")
    ax2[0,0].set_xlabel('Iteration')
    ax2[0,0].set_ylabel('Offset')
    ax2[0,0].set_title('Offset')
    ax2[0,0].legend()
    ax2[0,0].spines[['right', 'top']].set_visible(False)

    if color is not None:
        ax2[0,1].plot(df_freq['Iteration'], df_freq['frequency_flex'], color=color)
        ax2[0,1].plot(df_freq['Iteration'], df_freq['frequency_ext'], "--", color=color)
    else:
        ax2[0,1].plot(df_freq['Iteration'], df_freq['frequency_flex'])
        ax2[0,1].plot(df_freq['Iteration'], df_freq['frequency_ext'], "--")
    ax2[0,1].set_xlabel('Iteration')
    ax2[0,1].set_ylabel('Frequency (Hz)')
    ax2[0,1].set_title('Frequency')
    ax2[0,1].spines[['right', 'top']].set_visible(False)

    if color is not None:
        ax2[1,0].plot(df_amp['Iteration'], df_amp['amplitude_flex'], color=color)
        ax2[1,0].plot(df_amp['Iteration'], df_amp['amplitude_ext'], "--", color=color)
    else:
        ax2[1,0].plot(df_amp['Iteration'], df_amp['amplitude_flex'])
        ax2[1,0].plot(df_amp['Iteration'], df_amp['amplitude_ext'], "--")
    ax2[1,0].set_xlabel('Iteration')
    ax2[1,0].set_ylabel('Amplitude')
    ax2[1,0].set_title('Amplitude')
    ax2[1,0].spines[['right', 'top']].set_visible(False)

    if color is not None:
        ax2[1,1].plot(df_phase['Iteration'], df_phase['phase_flex'], color=color)
        ax2[1,1].plot(df_phase['Iteration'], df_phase['phase_ext'], "--", color=color)
    else:
        ax2[1,1].plot(df_phase['Iteration'], df_phase['phase_flex'])
        ax2[1,1].plot(df_phase['Iteration'], df_phase['phase_ext'], "--")
    ax2[1,1].set_xlabel('Iteration')
    ax2[1,1].set_ylabel('Phase')
    ax2[1,1].set_title('Phase')
    ax2[1,1].spines[['right', 'top']].set_visible(False)
    plt.tight_layout()

