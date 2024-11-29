""" Plot target and optimisation kinematics together """

import convert_data
import numpy as np
from net_osim import *
import opensim_sto_reader as os_read
import matplotlib.pylab as pylab
from colour import Color
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap

# plot params 
params = {'legend.fontsize': 20,
         'axes.labelsize': 20,
         'axes.titlesize':24, 
         'xtick.labelsize':20,
         'ytick.labelsize':20}
pylab.rcParams.update(params)


def main():

    # traj target
    traj = "circle"  # "flexion" or "circle"
    condition = 'optimisation_bi' # 'optimisation_bi', 'optimisation_w', 'movement' or 'perturbation'
    optim_algo = "cma" # "cma" "no_optim"
    with_sc = True  # True to plot the minimal SC scenario in condition ='optimisation_bi'

    if traj == "flexion":
        target_traj = "flexion_slow_2"
        period = 2.8  # in sec     
    elif traj == "circle":
        target_traj = "circles_mod_4"
        period = 1.3  # in sec 

    scenarios = [['baseline_1_'+target_traj, 'baseline_2_'+target_traj, 'baseline_3_'+target_traj],
                    ['magnitude_x0.5', 'baseline_1_' + target_traj, 'magnitude_x1.5','magnitude_x2.0'],
                    ['direction_-45', 'baseline_1_' + target_traj,'direction_45', 'direction_90']]
            


    # settings
    step_size = 0.01
    weights = np.arange(0, 1.1, 0.1)
    perturb_time = 2*period/5 
    show_plot = True

    # Extract scenarios data
    for subscenarios in scenarios:
        
        target_folder = "data/UP002/" + target_traj.split("_")[0] + "/" + target_traj.split("_")[1]+ "/" +  target_traj.split("_")[2] + "/" 
        
        target_files = [target_folder + target_traj.split("_")[0]+"_" + target_traj.split("_")[1]+"_"+target_traj.split("_")[2]+"_position.txt"] #,
                        #"data/UP002/flexion/mod/1/flexion_mod_1_position.txt",
                        #"data/UP002/flexion/mod/3/flexion_mod_3_position.txt"]
    
        files_time_emg = [target_folder + "emg.txt"] #,
                          #"data/UP002/flexion/mod/1/emg.txt",
                          #"data/UP002/flexion/mod/3/emg.txt"]  

        if "optimisation" in condition and "magnitude_x0.5" in scenarios:
            colors = ["salmon", "red", "tab:red", "darkred"]
        elif "optimisation" in condition:
            colors=["tab:blue", "tab:orange", "tab:green", "tab:red"]
        else:
            #colors=[colorFader("pink","maroon",i/len(weights)) for i in range(len(weights))]
            cmap = mpl.colormaps['plasma']
            colors = cmap(np.linspace(1, 0, 11))

        if condition == "optimisation_bi" and with_sc:
            input_folder = "brain_input_SC"
        elif condition == "optimisation_bi":
            input_folder = "brain_input"
        elif condition == "optimisation_w":
            input_folder = "SC_weights"
        else:
            if with_sc:
                input_folder = "control_input_sc/"
            else:
                input_folder = "control_input_nosc/"
        
        # SC pathways
        subfolder_fixed = ['fixed_/']
        subfolder_pathways = ['Ia_Mn', 'Ia_In', 'Ia_Mn_syn', 'Ib','II', 'Rn_Mn'] 

        # Extract data
        if 'optimisation' in condition:

            # traj results
            
            fig, ax = plt.subplots(1,2, figsize=(10, 5))
            fig2, ax2 = plt.subplots(1,2, figsize=(10, 5))
            plot_target = True
            g=0
            save_path = "results/" + traj +"/optimisation/" + input_folder + "/" + optim_algo + "/"
            for gravity in subscenarios:   
                position_file = "results/" + traj + "/optimisation/" + input_folder + "/" + optim_algo + "/" + gravity + "/simulation_States.sto"
                OS=os_read.readMotionFile(position_file)
                labels=OS[1]
                values=OS[2]
                n_lines=len(values)
                n_col=len(labels)
                position=[0]*n_lines
                sh_position=[0]*n_lines
                vel=[0]*n_lines
                sh_vel=[0]*n_lines
                time=[0]*n_lines

                for i in range (n_col):
                    if labels[i]=="/jointset/elbow/r_elbow_flexion/value":
                        for j in range(n_lines):
                            data=values[j]
                            position[j]=data[i]
                            time[j]=data[0]
                    elif labels[i]=="/jointset/shoulder/r_shoulder_elev/value":
                        for j in range(n_lines):
                            data=values[j]
                            sh_position[j]=data[i]
                    elif labels[i]=="/jointset/elbow/r_elbow_flexion/speed":
                        for j in range(n_lines):
                            data=values[j]
                            vel[j]=data[i]
                    elif labels[i]=="/jointset/shoulder/r_shoulder_elev/speed":
                        for j in range(n_lines):
                            data=values[j]
                            sh_vel[j]=data[i]

                data_f = np.column_stack([time, position, sh_position, vel, sh_vel])
                datafile_path = "results/" + traj + "/optimisation/" + input_folder + "/" + optim_algo + "/" + gravity + "/datafile.txt"
                np.savetxt(datafile_path , data_f, fmt='%f')

                # Plot trajectory
                scenario = "optimisation_" + gravity
                if "baseline_mod_1" not in subscenarios:
                    convert_data.convertData_pos_time_targ(ax, ax2, datafile_path, target_files[0], files_time_emg[0], scenario, plot_target, colors[g])
                else:
                    convert_data.convertData_pos_time_targ(ax, ax2, datafile_path, target_files[g], files_time_emg[g], scenario, plot_target, colors[g])
                g += 1
                if plot_target:
                    plot_target=False
            
            plt.tight_layout()
            fig.savefig(save_path + "traj_" + gravity + ".png")
            fig.savefig(save_path + "traj_" + gravity + ".svg")
                

    # Looping over the files in movement
    if 'optimisation' not in condition:

        # Set the fixed base directory
        base_directory = "results/" + traj + "/" + condition + "/" + input_folder

        for dir in subfolder_fixed:
            current_dir=base_directory+dir
            for pathway in subfolder_pathways:
                current_dir_p=current_dir+pathway+'/'
                fig, ax = plt.subplots(1,2, figsize=(10, 5))
                fig2, ax2 = plt.subplots(1,2, figsize=(10, 5))
                c = 0
                for w in weights:
                    current_dir_w=current_dir_p+pathway+'_'+str(int(w*10))+'/'
                    position_file=current_dir_w+'simulation_States.sto'
                    OS=os_read.readMotionFile(position_file)
                    labels=OS[1]
                    values=OS[2]
                    n_lines=len(values)
                    n_col=len(labels)
                    position=[0]*n_lines
                    sh_position=[0]*n_lines
                    vel=[0]*n_lines
                    sh_vel=[0]*n_lines
                    time=[0]*n_lines
                    for i in range (n_col):
                        if labels[i]=="/jointset/elbow/r_elbow_flexion/value":
                            for j in range(n_lines):
                                data=values[j]
                                position[j]=data[i]
                                time[j]=data[0]
                        elif labels[i]=="/jointset/shoulder/r_shoulder_elev/value":
                            for j in range(n_lines):
                                data=values[j]
                                sh_position[j]=data[i]
                        elif labels[i]=="/jointset/elbow/r_elbow_flexion/speed":
                            for j in range(n_lines):
                                data=values[j]
                                vel[j]=data[i]
                        elif labels[i]=="/jointset/shoulder/r_shoulder_elev/speed":
                            for j in range(n_lines):
                                data=values[j]
                                sh_vel[j]=data[i]

                    data_f = np.column_stack([time, position, sh_position, vel, sh_vel])
                    datafile_path = current_dir_w+'datafile.txt'
                    np.savetxt(datafile_path , data_f, fmt='%f')

                    scenario = pathway + "_" + str(round(w,1))
                    if condition == "perturbation":
                        plot_target=False
                    else:
                        if w == 0:
                            plot_target=True
                        else:
                            plot_target=False
                    convert_data.convertData_pos_time_targ(ax, ax2, datafile_path, target_files[0], files_time_emg[0], scenario, plot_target, color=colors[c])
                    c += 1
                    
                if condition == "perturbation":
                    ymin, ymax = ax[1].get_ylim()
                    ax[1].axvline(perturb_time+step_size, ymin, ymax, color="grey", linestyle="--")
                    ymin, ymax = ax[0].get_ylim()
                    ax[0].axvline(perturb_time+step_size, ymin, ymax, color="grey", linestyle="--")

                norm = BoundaryNorm(weights, cmap.N)
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[0])
                fig2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2[0])
                plt.tight_layout()
                fig.savefig(current_dir_p+'targ_pos_plot')
                fig.savefig(current_dir_p+'targ_pos_plot.svg')
                fig2.savefig(current_dir_p+'vel_plot')
                fig2.savefig(current_dir_p+'vel_plot.svg')
                plt.close()

    if show_plot:
        plt.show()


def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


if __name__ == '__main__':
    main()
