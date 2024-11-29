"""functions to integrate and simulate SC networks with OpenSim models """

import opensim
from opensim_environment import OsimModel
from gen_net import *


def net_osim(osim_file, step_size, n_steps, nn_file_name, net_model_file, include, controls_dict, save_folder,
             nn_init=None, integ_acc=0.0001, perturb=None, gravity=[0, -9.81, 0], markers=None, coord_plot=None, musc_acti=None, rates_plot=None, visualize=False, show_plot=False, save_kin=None):
    """ Integrate and simulate network and osim models """

    #: Osim model
    if perturb is not None:
        model = OsimModel(osim_file, step_size, integ_acc, perturb['body'], visualize=visualize, save_kin=save_kin)
    else:
        model = OsimModel(osim_file, step_size, integ_acc, perturb, visualize=visualize, save_kin=save_kin)
    model.model.setGravity(opensim.Vec3(gravity[0], gravity[1], gravity[2]))
    muscles = model.model.getMuscles()
    n_muscles = muscles.getSize()
    muscle_names = [None]*n_muscles
    for i in range(n_muscles):
        muscle_names[i] = muscles.get(i).getName()

    #: Descending controls
    controls = all_controls(controls_dict, muscle_names, n_steps)

    #: SC model
    sc_model = build_sc_xlsx(net_model_file, step_size)
    for j in sc_model.ext_muscles.keys():
        sc_model.ext_muscles[j].update(sc_model.flex_muscles[j])
    if len(sc_model.ext_muscles.keys()) == 1:
        sc_model.ext_muscles = sc_model.ext_muscles[list(sc_model.ext_muscles.keys())[0]]
    else:
        for j in range(1, len(sc_model.ext_muscles.keys())):
            sc_model.ext_muscles[list(sc_model.ext_muscles.keys())[0]].update(
                sc_model.ext_muscles[list(sc_model.ext_muscles.keys())[j]])
        sc_model.ext_muscles = sc_model.ext_muscles[list(sc_model.ext_muscles.keys())[0]]

    #: Initialize network
    container = Container(max_iterations=100000)
    net_ = NeuralSystem(
        os.path.join(
            os.path.dirname(__file__),
            save_folder+nn_file_name+'.graphml'
        ),
        container
    )
    container.initialize()
    if nn_init is not None:
        net_.setup_integrator(nn_init*np.ones(len(container.neural.states.names)))
    else:
        net_.setup_integrator()

    #: Initialise osim model
    model.reset()
    model.reset_manager()

    #: Actuate and integrate step 0
    model.actuate(np.zeros(n_muscles))
    model.integrate()

    #: States to store
    if markers is not None:
        marker_pos = np.zeros((len(markers), 3, n_steps-1))
        marker_vel = np.zeros((len(markers), 3, n_steps-1))
    if coord_plot is not None:
        coord_states = np.zeros((len(coord_plot), 2, n_steps-1))
    if musc_acti is not None:
        if musc_acti == "all":
            musc_acti = muscle_names 
        musc_states = np.zeros((len(musc_acti), n_steps-1))
    if rates_plot is not None:
        if rates_plot[0] == 'all':
            rates_plot[0] = muscle_names
        rates = np.zeros((len(rates_plot[0]), len(rates_plot[1]), n_steps-1))

    #: Actuate and integrate next steps
    mn_rates = np.zeros(n_muscles)
    p = 0
  
    for j in range(1, n_steps):
        #: Integrate network with descending and Ia rates
        for i in range(n_muscles):
            muscle = model.model.getMuscles().get(i)  # OpenSim
            # controls
            container.neural.inputs.get_parameter('aff_arm_' + muscle_names[i] + '_C').value = controls["Mn"][i, j]
            # Ia rates
            if "Ia" in include:
                sc_model.ext_muscles[muscle_names[i]].Prochazka_Ia_rates(model, muscle)
                container.neural.inputs.get_parameter('aff_arm_' + muscle_names[i] + '_Ia').value = \
                            sc_model.ext_muscles[muscle_names[i]].past_Ia_rates[0]
            # II rates
            if 'II' in include:
                sc_model.ext_muscles[muscle_names[i]].Prochazka_II_rates(model, muscle)
                container.neural.inputs.get_parameter('aff_arm_' + muscle_names[i] + '_II').value = \
                    sc_model.ext_muscles[muscle_names[i]].past_II_rates[0]
             # II rates
            if 'Ib' in include:
                sc_model.ext_muscles[muscle_names[i]].Prochazka_Ib_rates(model, muscle)
                container.neural.inputs.get_parameter('aff_arm_' + muscle_names[i] + '_Ib').value = \
                    sc_model.ext_muscles[muscle_names[i]].past_Ib_rates[0]                   

        net_.step(dt=step_size)
        for i in range(n_muscles):
            mn_rates[i] = container.neural.outputs.get_parameter('nout_arm_Mn_' + muscle_names[i]).value

        # Pertubation
        if perturb is not None:
            force = perturb["force"]
            dir = perturb['dir']
            onset = np.array(perturb["onset"])/step_size
            delta_t = perturb["delta_t"]
            if p < len(force):
                ext_force = opensim.PrescribedForce.safeDownCast(model.model.getForceSet().get('perturbation'))
                forceFunctionSet = ext_force.get_forceFunctions()
                func = opensim.Constant.safeDownCast(forceFunctionSet.get(dir[p]))
                if j in np.arange(onset[p], onset[p] + delta_t + 1, 1):
                    func.setValue(force[p])
                    if j == onset[p] + delta_t:
                        p += 1
                        func.setValue(0)
                else:
                    func.setValue(0)
            else:
                if j > onset[p-1] + delta_t:
                    func.setValue(0)

        #: Actuate the model from control outputs
        model.actuate(mn_rates)

        #: Integration musculoskeletal system
        model.integrate()

        #: States to store
        res = model.get_state_dict()
        if markers is not None:
            for i in range(len(markers)):
                marker_pos[i, :, j - 1] = res["markers"][markers[i]]["pos"]
                marker_vel[i, :, j - 1] = res["markers"][markers[i]]["vel"]

        if coord_plot is not None:
            for i in range(len(coord_plot)):
                coord_states[i, 0, j-1] = res["coordinate_pos"][coord_plot[i]]
                coord_states[i, 1, j-1] = res["coordinate_vel"][coord_plot[i]]
        
        if musc_acti is not None:
            for m in range(len(musc_acti)):
                musc_states[m, j-1] = res['muscles'][musc_acti[m]]['activation']

        if rates_plot is not None:
            for m in range(len(rates_plot[0])):
                for r in range(len(rates_plot[1])):
                    if rates_plot[1][r] == 'Mn':
                        rates[m, r, j-1] = mn_rates[muscle_names.index(rates_plot[0][m])]
                    elif rates_plot[1][r] == 'Ia':
                        rates[m, r, j-1] = net_.container.neural.inputs.values[
                            net_.container.neural.inputs.names.index('aff_arm_'+rates_plot[0][m]+'_'+"Ia")]
                    elif rates_plot[1][r] == 'II':
                        rates[m, r, j-1] = net_.container.neural.inputs.values[
                            net_.container.neural.inputs.names.index('aff_arm_'+rates_plot[0][m]+'_'+"II")]
                    elif rates_plot[1][r] == 'Ib':
                        rates[m, r, j-1] = net_.container.neural.inputs.values[
                            net_.container.neural.inputs.names.index('aff_arm_'+rates_plot[0][m]+'_'+"Ib")]
                    else:
                        prefix = net_.container.neural.states.names[0][0]
                        rates[m, r, j-1] = net_.container.neural.states.values[
                            net_.container.neural.states.names.index(prefix+'_arm_'+rates_plot[1][r]+'_'+rates_plot[0][m])]
    # Control plot
    time = np.arange(n_steps) * step_size
    keys = list(controls_dict["Mn"].keys())
    idx_control = [muscle_names.index(muscle) for muscle in keys]
    plt.figure("Control plot")
    fig, axes = plt.subplots(nrows=1, ncols=len(keys), figsize=(10,5))
    for i, idx in enumerate(idx_control):
        col = idx % len(keys)
        ax = axes[col]
        ax.plot(time, controls['Mn'][idx])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Control input")
        ax.set_ylim((0,1))
        ax.set_title("Mn" + muscle_names[idx])
    plt.savefig(save_folder + "control plot")

    #: Markers plot
    time = np.arange(n_steps - 1) * step_size
    if markers is not None:
        plt.figure("marker plot")
        for i in range(len(markers)):
            # Y pos and vel
            ax = plt.subplot(3, 1, 1)
            lns1 = ax.plot(time, marker_pos[i, 1] * 100, 'b', label="pos_Y")
            ax2 = ax.twinx()
            lns2 = ax2.plot(time, marker_vel[i, 1], 'c', label="vel_Y")
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("pos [cm]", color='b')
            ax2.set_ylabel("vel [m/s]", color='c')
            ax.set_title("Hand Y kinematics")

            # X pos and vel
            ax = plt.subplot(3, 1, 2)
            lns1 = ax.plot(time, marker_pos[i, 0] * 100, 'b', label="pos_X")
            ax2 = ax.twinx()
            lns2 = ax2.plot(time, marker_vel[i, 0], 'c', label="vel_X")
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_xlabel("time [s]")
            ax.set_ylabel("pos [cm]", color='b')
            ax2.set_ylabel("vel [m/s]", color='c')
            ax.set_title("Hand X kinematics")

            # vel
            marker_v = np.sqrt(np.power(marker_vel[i, 0], 2) + np.power(marker_vel[i, 1], 2))
            ax = plt.subplot(3, 1, 3)
            ax.plot(time, marker_v, 'r', label="vel")
            ax.legend()
            ax.set_xlabel("time [s]")
            ax.set_ylabel("vel [m/s]")
            ax.set_title("Hand vel")
            plt.savefig(save_folder + "marker_plot")

    #: Coordinates plot
    if coord_plot is not None:
        plt.figure("coord plot")
        for i in range(len(coord_plot)):
            ax = plt.subplot(len(coord_plot), 1, i + 1)
            lns1 = ax.plot(time, coord_states[i, 0], 'b', label="pos", )
            ax2 = ax.twinx()
            lns2 = ax2.plot(time, coord_states[i, 1], 'c', label="vel")
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs)
            ax.set_ylabel("angle [rad]", color='b')
            ax.set_ylim((0,3))
            ax2.set_ylabel("vel [rad/s]", color='c')
            ax.set_xlabel("time [s]")
            plt.title(coord_plot[i] + " kinematics")
        plt.savefig(save_folder + "coord_plot")

    #: Muscles acti plot
    if musc_acti is not None:
        plt.figure("muscle acti plot")
        for m in range(len(musc_acti)):
            plt.plot(time, musc_states[m,:], label=musc_acti[m])
            plt.legend()
            plt.ylabel("muscle activation")
            plt.xlabel("time [s]")
            plt.title("Muscles activation")
        plt.savefig(save_folder + "muscle_acti")

    #: Rates plot
    if rates_plot is not None:
        fig = plt.figure('rates plot')
        if len(rates_plot[1]) == 1:
            for m in range(len(rates_plot[0])):
                plt.plot(time, rates[m, 0], label=rates_plot[0][m])
                plt.ylabel("firing rate")
                plt.title(rates_plot[1][0] + " firing rates")
                plt.xlabel("time [s]")
                plt.legend()
        else:
            axes = fig.subplots(nrows=len(rates_plot[1]), ncols=1)
            for r in range(len(rates_plot[1])):
                for m in range(len(rates_plot[0])):
                    axes[r].plot(time, rates[m, r], label=rates_plot[0][m])
                axes[r].set_ylabel("firing rate")
                axes[r].set_title(rates_plot[1][r] + " firing rates")
                axes[-1].set_xlabel("time [s]")
                lines, labels = fig.axes[-1].get_legend_handles_labels()
                fig.legend(lines, labels)
        fig.tight_layout()
        fig.savefig(save_folder + "rates_plot")

    if show_plot:
        plt.show()
    plt.close('all')

    #: Save kinematics
    if save_kin is not None:
        model.save_simulation(save_folder)

    return coord_states[0,0,:], coord_states[1,0,:]  # shoulder, elbow as given in coord_plot 


    
def all_controls(controls_dict, muscle_names, n_steps):
    """ Descending controls """

    n_muscles = len(muscle_names)
    controls = dict()
    # Mn controls
    controls["Mn"] = np.zeros((n_muscles, n_steps))
    keys = list(controls_dict["Mn"].keys())
    for k in keys:
        controls["Mn"][muscle_names.index(k)] = np.concatenate((controls_dict["Mn"][k], np.zeros(n_steps-len(controls_dict["Mn"][k]))))
    
    # Ia controls
    if "Ia" in controls_dict.keys():
        controls["Ia"] = np.zeros((n_muscles, n_steps))
        keys = list(controls_dict["Ia"].keys())
        for k in keys:
            controls["Ia"][muscle_names.index(k)] = np.concatenate((controls_dict["Ia"][k], np.zeros(n_steps-len(controls_dict["Ia"][k]))))

    return controls
