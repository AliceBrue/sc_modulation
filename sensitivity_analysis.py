""" Perform sensitivity analysis for the various SC pathways 
    on movement smoothness and deviation to perturbation 
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
from run_net_sim_groups_input import *
from metrics_osim_input_rmse import *


def main(): 

    # set scenario
    traj = "circle"  # "flexion" or "circle"
    condition = "perturb"  # "smoothness" or "perturb"
    calc_second_order = True
    N = 512 
    run_from = None  ##
    run_sim  = False
    run_analyse = True

    if traj == "flexion":
        target_traj = "1D_flexion_slow_2" 
        traj_period = 2.8  # in sec
    elif traj == "circle":
        target_traj = "circles_mod_4" 
        traj_period = 1.3  # in sec

    problem = {'num_vars': 6,
                'names': ['Ia_Mn', 'Ia_In', 'Ia_Mn_syn', 'Ib', 'II', 'Rn_Mn'],
                'bounds': [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]}
    
    save_folder = "results/"+traj+"/sensitivity/"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    save_folder += condition + "/"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    if run_sim:

        if run_from is None:
            param_values = saltelli.sample(problem, N, calc_second_order=calc_second_order)
            np.savetxt(save_folder + "/param_values_" + str(N) +".txt", param_values)
        
        param_values = np.loadtxt("results/sensitivityc/" + condition + "/param_values_" + str(N) +".txt", float)

        if condition == "perturb":
            metric_list = ["deviation"]
        else:
            metric_list = ["sal", "rmse"]
        
        Y = np.zeros((param_values.shape[0]))
        res_files = [save_folder + "outputs_" + str(N) + "_" + metric_list[0] +".txt"]
        if len(metric_list) > 1 :
            for m in metric_list[1:]:
                res_files.append(save_folder + "outputs_" + str(N) + "_" + m +".txt")
        for i, X in enumerate(param_values):
            if run_from is None: 
                print("SIM", i)
                Y[i] = evaluate_model(X, target_traj, traj_period, metric_list, res_files)
            elif i >= run_from:
                print("SIM", i)
                Y[i] = evaluate_model(X, target_traj, traj_period, metric_list, res_files)
            plt.close("all")

    if run_analyse:

        if condition == "perturb":
            metric_list = ["deviation"]
        else:
            metric_list = ["sal", "rmse"]

        for m in range(len(metric_list)):
            Y = np.loadtxt("results/" + traj + "/sensitivity/" + condition + "/outputs_" + str(N) + "_" + metric_list[m] + ".txt", float)
            Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order)
        
            # save text
            with open("results/" + traj + "/sensitivity/" + condition + "/si_res_" + metric_list[m] + ".txt", 'w') as f:  
                for key, value in Si.items():  
                    f.write('%s:%s\n' % (key, value))
            
            # plot
            Si.plot()
            plt.savefig("results/" + traj + "/sensitivity/" + condition + "/si_plot_" + metric_list[m] + ".svg")
            plt.show()


def evaluate_model(X, target_traj, sim_period, metric_list, res_files):

    # target traj
    target_folder = "data_1D_flexion/UP002/circles/" + target_traj.split("_")[1]+ "/" +  target_traj.split("_")[2]+ "/" 
    target_file = target_folder + "circles_"+target_traj.split("_")[1]+"_"+target_traj.split("_")[2]+"_position.txt"

    # brain input
    optim_folder = "brain_input_SC/cma"
    optim_folder = "results/optimisation_circles/" + optim_folder + "/baseline_1_" + target_traj+"/"
    params_flex, params_ext = read_optim_param(optim_folder)
    print("Brain input param: ", params_flex, params_ext)

    if "deviation" in metric_list: 
        save_folder = 'results/sensitivityc/perturb/'
    else:
        save_folder = 'results/sensitivityc/smoothness/'
    
    # osim
    visualize = False
    show_plot = False
    osim_file = 'models/full_arm_M_delt1_wrap.osim'
    n_muscles = 7
    osim_file = lock_Coord(osim_file, ['r_shoulder_elev'], 'false')  # "true" if you want to lock the shoulder
    osim_file = lock_Coord(osim_file, ['elv_angle', 'shoulder_rot'], 'true')  # "false" if you want to unlock the shoulder in 3D
    target = pd.read_csv(target_file, sep=' ')
    sh_target0 = target.iloc[:, 0].values[0]
    elb_target0 = target.iloc[:, 1].values[0]
    modify_default_Coord(osim_file, 'r_shoulder_elev', sh_target0)
    modify_default_Coord(osim_file, 'r_elbow_flexion', elb_target0)

    #:Osim settings
    step_size = 0.01
    integ_acc = 0.0001
    # Simulation period
    n_steps = int(sim_period/step_size)    
    time_points = np.linspace(0, sim_period, n_steps)

    #: Perturbation
    if "deviation" in metric_list: 
        perturb_time = 2*sim_period/5
        perturb = {'body': 'r_hand', 'force': [-30], 'dir': [1], 'onset': [perturb_time], 'delta_t': 30} 
    else: 
        perturb_time = 0
        perturb = None
        
    #: Network from sc model
    nn_file_name = 'net'
    #gen_net_file('models/net_model', Ia_w=1, IaIn_w=0.5, II_w=0.5)  # generate a sc model file with new reflex parameters
    neuron_model = 'leaky'  # 'lif_danner' more bio model but more param to definem check FARMS implementation
    tau = 0.001 # leaky time constant
    bias = -0.5 # center sigmoid to 0.5
    D = 8 # sigmoid slope at bias
    nn_init = 0  # -60 for lif_dannernet_osim


    #Delay before controls and period of muscle activation
    delay = 0  # delay before controls in sec
    delay = int(delay/step_size)

    # Net model parameters
    columns         = ['Muscles','type','Ia_antagonist','Ia_synergistic','Ia_delay','Ia_Mn_w','Ia_In_w','IaIn_Mn_w','IaIn_IaIn_w','Mn_Rn_w',
                       'Rn_Mn_w','Rn_IaIn_w','Rn_Rn_w','II_delay','II_w','Ib_delay','Ib_w','IbIn_w', 'Ia_Mn_w_syn'] 
    muscle_list     = ['DELT1', 'TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA']
    type_list       = ['elb_flex','elb_ext','elb_ext','elb_ext','elb_flex','elb_flex','elb_flex']
    antagonist_list = ['TRIlong','DELT1,BIClong','BICshort,BRA','BICshort,BRA','TRIlong','TRIlat,TRImed','TRIlat,TRImed']
    synergistic_list = ['BIClong', 'TRIlat,TRImed', 'TRIlong,TRImed', 'TRIlong,TRIlat', 'BICshort,BRA,DELT1', 'BIClong,BRA', 'BIClong,BICshort'] #or None
    null_column = np.zeros(n_muscles)
    net = pd.DataFrame(zip(muscle_list,type_list,antagonist_list,synergistic_list,null_column,null_column,null_column,null_column,null_column,
                           null_column,null_column,null_column,null_column,null_column,null_column,null_column,
                           null_column,null_column,null_column), columns=columns) 
    
    # build net from sensitivity weights
    pathways = ['Ia_Mn', 'Ia_In', 'Ia_Mn_syn', 'Ib', 'II', 'Rn_Mn']

    net_model = net.copy()
    net_model,include1,_, rates_plot = select_connections(pathways[0], net_model, round(X[0], 2), save_group=False)
    net_model2,include2,_, rates_plot = select_connections(pathways[1], net_model, round(X[1], 2), save_group=False) ##need to add connections based on the length of fixed_pathways
    net_model3,include3,_, rates_plot = select_connections(pathways[2], net_model2, round(X[2], 2), save_group=False)
    net_model4,include4,_, rates_plot = select_connections(pathways[3], net_model3, round(X[3], 2), save_group=False)
    net_model5,include5,_, rates_plot = select_connections(pathways[4], net_model4, round(X[4], 2), save_group=False)
    net_model6,include6,_, rates_plot = select_connections(pathways[5], net_model5, round(X[5], 2), save_group=False)
    include = include1 + include2 + include3 + include4 + include5 + include6
    include =  list(set(include))

    scenario = str(round(X[0], 2)) + "_" +  str(round(X[1],2)) + "_" + str(round(X[2], 2)) + "_" + str(round(X[3], 2)) + "_" + str(round(X[4], 2)) + "_" + str(round(X[5], 2)) 

    current_save = save_folder
    if not os.path.isdir(current_save):
        os.makedirs(current_save, exist_ok=True)

    net_model6.to_excel(current_save + 'net_model_current_scenario.xlsx')          
            
    # create excel file in each subfolder 
    nn_file_name = 'net_model_current_scenario'
    net_model_file = os.path.join(current_save, nn_file_name + '.xlsx')
        
    #: plots
    markers = ["r_hand_com"]
    coord_plot = ['r_shoulder_elev', 'r_elbow_flexion']
    musc_acti = "all"
    
    gen_net('arm', nn_file_name, net_model_file, step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=include,
                    save_folder=current_save, legend=True, show_plot=show_plot)

    #: Simulation
    save_kin = True 

    #for control in controls : 
    controls_dict_in = brain_inputs.brain_inputs(params_flex, params_ext, time_points)  
    controls_dict = dict()
    controls_dict["Mn"] = dict()
    controls_dict["Mn"]["BIClong"] = controls_dict_in["signal_delt"]
    controls_dict["Mn"]["BICshort"] = controls_dict_in["signal_values_flex"]
    controls_dict["Mn"]["BRA"] = controls_dict_in["signal_values_flex"]
    controls_dict["Mn"]["DELT1"] = controls_dict_in["signal_delt"]

    controls_dict["Mn"]["TRIlong"] = controls_dict_in["signal_values_ext"]
    controls_dict["Mn"]["TRIlat"] = controls_dict_in["signal_values_ext"]
    controls_dict["Mn"]["TRImed"] = controls_dict_in["signal_values_ext"]

    net_osim(osim_file, step_size, n_steps, nn_file_name, net_model_file, include, controls_dict, nn_init=nn_init, 
                        integ_acc=integ_acc, perturb=perturb, markers=markers, coord_plot=coord_plot, musc_acti=musc_acti, 
                        rates_plot=rates_plot, save_folder=current_save, visualize=visualize, show_plot=show_plot, save_kin=save_kin)
    
    joint = "elbow"
    param_list = ['kinematics_pos', 'kinematics_vel', 'speed']  
    params = get_params(param_list, save_folder)
    time = params[0].get('time').to_numpy()
    
    # target traj
    joints_target = pd.read_csv(target_file, sep=' ')
    if joint == "shoulder":
        joint_target = joints_target.iloc[:, 0].values
    if joint == "elbow":
        joint_target = joints_target.iloc[:, 1].values

    metric_val, metric_name = compute_metrics(scenario, joint, metric_list, params, save_folder, time, step_size, perturb_time, delay, perturb, w_neg=0.5, joint_target=joint_target)

    # metric indexes
    m_ind = np.zeros(len(metric_list))
    for m in range(len(metric_list)):
        m_ind[m] = int(metric_name.index(metric_list[m]))
    m_ind = m_ind.astype('int64')

    for m in range(len(metric_list)):
        with open(res_files[m],'a') as f:
            f.write(str(metric_val[m_ind[m]]) + "\n")

    return metric_val[m_ind[0]]
   

if __name__ == '__main__':
    main()
