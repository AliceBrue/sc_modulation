#plot_input

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
import convert_data

# plot params
params = {'legend.fontsize': 20,
         'axes.labelsize': 20,
         'axes.titlesize':24,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
pylab.rcParams.update(params)


def main():

    # Set scenario
    traj = "circle"  # "flexion" or "circle"
    optimization_algo = "cma"  # "pso", "cma", "de"
    with_sc = True  # True for the minimal SC scenario
    show_plot = True

    base_directory = "results/" + traj + "/optimisation/brain_input"
    if with_sc: 
        base_directory += "_SC/" + optimization_algo + "/"
    
    if traj == "flexion":
        target_traj = "flexion_slow_2"
        period = 2.8  # in sec
    elif traj == "circle":
        target_traj = "circles_mod_4"
        period = 1.3  # in sec

    scenarios=[['baseline_1_' + target_traj,'baseline_2_' + target_traj, 'baseline_3_' + target_traj], 
                ['magnitude_x0.5', 'baseline_1_' + target_traj, 'magnitude_x1.5','magnitude_x2.0'],
                ['direction_-45', 'baseline_1_' + target_traj,'direction_45', 'direction_90']]

    sim_period = np.array([period, 1.85, 1.7]) # in sec
    step_size = 0.01
    n_steps = sim_period/step_size
    n_steps = n_steps.astype(int)
    time = [np.linspace(0, sim_period[0], n_steps[0]), np.linspace(0, sim_period[1], n_steps[1]), np.linspace(0, sim_period[2], n_steps[2])]


    for subscenarios in scenarios:
        # plot params
        label_fe = True
            
        parameters = {} 
        fig, ax = plt.subplots(1,1, figsize=(10, 5))
        fig2, ax2 = plt.subplots(2,2, figsize=(15,12))

        if "magnitude_x0.5" in subscenarios:
            colors = ["salmon", "red", "tab:red", "darkred"]
        else:
            colors=["tab:blue", "tab:orange", "tab:green", "tab:red"]

        c = 0
        for dir in subscenarios:
            
            current_dir = base_directory + dir + '/'
            params_file = current_dir + '_best_results_input.txt'

            convert_data.plot_params(ax, ax2, params_file, colors[c], label=dir, label_fe=label_fe)
            label_fe = False
            
            with open(params_file, "r") as f_p:
                lines_param = f_p.readlines()

            # Store the last set of parameters for each subfolder
            last_line = lines_param[-1].split(', ')
            parameters[dir] = {
                'loss': last_line[1],
                'offset_flex': float(last_line[2]),
                'frequency_flex': float(last_line[3]),
                'amplitude_flex': float(last_line[4]),
                'phase_flex': float(last_line[5]),
                'offset_ext': float(last_line[6]),
                'frequency_ext': float(last_line[7]),
                'amplitude_ext': float(last_line[8]),
                'phase_ext': float(last_line[9]),    
            }
            c += 1 

        fig.savefig(base_directory + "loss" + subscenarios[-1] + ".png")
        fig.savefig(base_directory + "loss" + subscenarios[-1] + ".svg")
        fig2.savefig(base_directory + "params" + subscenarios[-1]+ ".png")
        fig2.savefig(base_directory + "params" + subscenarios[-1] + ".svg")
            

        #Plot inputs
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        # flexors
        c = 0
        for dir, params in parameters.items():
            if "mod_1" not in subscenarios and "mod_3" not in subscenarios:
                signal_flex = params['offset_flex'] + params['amplitude_flex'] * np.sin(2 * np.pi * params['frequency_flex'] * time[0] + params['phase_flex'])
                if colors is not None:
                    ax[0].plot(time[0], signal_flex, color=colors[c], label=dir)
                else:
                    ax[0].plot(time[0], signal_flex, label=dir)
            else:
                signal_flex = params['offset_flex'] + params['amplitude_flex'] * np.sin(2 * np.pi * params['frequency_flex'] * time[c] + params['phase_flex'])
                if colors is not None:
                    ax[0].plot(time[c], signal_flex, color=colors[c], label=dir)
                else:
                    ax[0].plot(time[c], signal_flex, label=dir)
            c += 1

        ax[0].set_xlabel('Time (figs)')
        ax[0].set_ylabel('Input Flex')
        ax[0].set_ylim(0, 1)
        if len(subscenarios) >1:
            ax[0].legend()
        ax[0].set_title('Flexors input')
        ax[0].spines[['right', 'top']].set_visible(False)

        #extensors
        c = 0
        for dir, params in parameters.items():
            if "mod_1" not in subscenarios and "mod_3" not in subscenarios:
                signal_ext = params['offset_ext'] + params['amplitude_ext'] * np.sin(2 * np.pi * params['frequency_ext'] * time[0] + params['phase_ext'])
                if colors is not None:
                    ax[1].plot(time[0], signal_ext, color=colors[c], label=dir) #, linestyle = "--")
                else:
                    ax[1].plot(time[0], signal_ext, label=dir) #, linestyle = "--")
            else:
                signal_ext = params['offset_ext'] + params['amplitude_ext'] * np.sin(2 * np.pi * params['frequency_ext'] * time[c] + params['phase_ext'])
                if colors is not None:
                    ax[1].plot(time[c], signal_ext, color=colors[c], label=dir) #, linestyle = "--")
                else:
                    ax[1].plot(time[c], signal_ext, label=dir) #, linestyle = "--")
            c += 1

        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Input Ext')
        ax[1].set_ylim(0, 1)
        ax[1].set_title('Extensors input')
        ax[1].spines[['right', 'top']].set_visible(False)

        fig.tight_layout()
        if len(subscenarios) == 1 :
            fig.savefig(base_directory + subscenarios[0] + "/plot_input")
            fig.savefig(base_directory + subscenarios[0] + "/plot_input.svg")
        else:
            fig.savefig(base_directory + "plot_input" + subscenarios[-1]+ ".png")
            fig.savefig(base_directory + "plot_input" + subscenarios[-1] + ".svg")

    if show_plot:
        plt.show()


if __name__ == '__main__':
    main()
