""" Generate SC networks with various weights and simulate opensim model 
    Possible SC groups are: ['Ia_Mn', 'Ia_Mn_syn', 'Ia_In','Ib', 'II', 'Rn_Mn'] 
"""

from multiprocessing import current_process
from net_osim import *
from osim_model import *
from connections import *
from metrics import *
import brain_inputs

def main():

    # set scenario
    traj = "flexion"  # "flexion" or "circle"
    sim_perturb = True # True for perturbation scenario

    input_folder = "control_input_nosc"
    fixed_pathways= []  
    groups = ['Ia_Mn', 'Ia_In', 'Ia_Mn_syn', 'Ib', 'II', 'Rn_Mn'] 
    
    # Brain input 
    if traj == "flexion":
        target_traj = "flexion_slow_2"
        sim_period = 2.8  # in sec
    elif traj == "circle":
        target_traj = "circles_mod_4"
        sim_period = 1.3  # in sec
    optim_folder = "brain_input/cma"
    optim_folder = "results/" + traj + "/optimisation/" + optim_folder + "/baseline_1_" + target_traj+"/"
    params_flex, params_ext = read_optim_param(optim_folder)
    print("Brain input param: ", params_flex, params_ext)

    # Perturbation time
    perturb_time = 2*sim_period/5
    
    # target traj
    target_folder = "data/UP002/" + target_traj.split("_")[0] + "/" + target_traj.split("_")[1]+ "/" +  target_traj.split("_")[2] + "/" 
    target_file = target_folder + target_traj.split("_")[0]+"_" + target_traj.split("_")[1]+"_"+target_traj.split("_")[2]+"_position.txt"

    visualize = False
    show_plot = False
    
    #: Osim models
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
    
    # Brain input
    controls_dict_in = brain_inputs.brain_inputs(params_flex, params_ext, time_points)

    #: Results folder
    if sim_perturb:
        save_folder = 'results/' + traj + '/perturbation/'
    else :
        save_folder = 'results/' + traj + '/movement/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    save_folder += input_folder + "/"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    #: Network from sc model
    nn_file_name = 'net'
    #gen_net_file('models/net_model', Ia_w=1, IaIn_w=0.5, II_w=0.5)  # generate a sc model file with new reflex parameters
    neuron_model = 'leaky'  # 'lif_danner' more bio model but more param to definem check FARMS implementation
    tau = 0.001 # leaky time constant
    bias = -0.5 # center sigmoid to 0.5
    D = 8 # sigmoid slope at bias
    nn_init = 0  # -60 for lif_dannernet_osim
    weights =  np.arange(0, 1.1, 0.1)

    #Delay before controls and period of muscle activation
    delay = 0  # delay before controls in sec
    delay = int(delay/step_size)

    #: Perturbations
    perturb = None 
    if sim_perturb and traj == "flexion":
        perturb = {'body': 'r_hand', 'force': [30], 'dir': [1], 'onset': [perturb_time], 'delta_t': 30} 
    elif sim_perturb and traj == "circle":
        perturb = {'body': 'r_hand', 'force': [-30], 'dir': [1], 'onset': [perturb_time], 'delta_t': 30} 

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
    
    #for control in controls :   
    controls_dict = dict()
    controls_dict["Mn"] = dict()
    controls_dict["Mn"]["BIClong"] = controls_dict_in["signal_delt"]
    controls_dict["Mn"]["BICshort"] = controls_dict_in["signal_values_flex"]
    controls_dict["Mn"]["BRA"] = controls_dict_in["signal_values_flex"]
    controls_dict["Mn"]["DELT1"] = controls_dict_in["signal_delt"]

    controls_dict["Mn"]["TRIlong"] = controls_dict_in["signal_values_ext"]
    controls_dict["Mn"]["TRIlat"] = controls_dict_in["signal_values_ext"]
    controls_dict["Mn"]["TRImed"] = controls_dict_in["signal_values_ext"]

    #: Big loop to iteratively write and select from all groups and all weights the appropriate model
    if len(fixed_pathways)>=2:
        for group in groups:
            net_model = net.copy()
            for w in weights:
                net_model,include1,_, rates_plot = select_connections(fixed_pathways[0], net_model, fixed_weights[0], save_group=False)
                net_model2,include2,_, rates_plot = select_connections(fixed_pathways[1], net_model, fixed_weights[1], save_group=False) ##need to add connections based on the length of fixed_pathways
                net_model3,include3,_, rates_plot = select_connections(group, net_model2, w, save_group=False)
                include = include1 + include2 + include3

                if not os.path.isdir (save_folder + 'fixed_' + fixed_pathways[0] + '_' + fixed_pathways[1] +  '/'):
                    os.makedirs( save_folder + 'fixed_' + fixed_pathways[0] + '_' + fixed_pathways[1] +  '/')
                
                if not os.path.isdir(save_folder + 'fixed_' + fixed_pathways[0] + '_' + fixed_pathways[1] +  '/' + group + '/'):
                    os.makedirs(save_folder + 'fixed_' + fixed_pathways[0] + '_' + fixed_pathways[1] +  '/' + group + '/')

                if not os.path.isdir(save_folder + 'fixed_' + fixed_pathways[0] + '_' + fixed_pathways[1] +  '/' + group + '/' + group + '_'  +  str(int(w*10)) + '/'):
                    os.makedirs(save_folder + 'fixed_' + fixed_pathways[0] + '_' + fixed_pathways[1] +  '/' + group + '/' + group + '_'  +  str(int(w*10)) + '/')

                current_save = save_folder + 'fixed_' + fixed_pathways[0] + '_' + fixed_pathways[1] +  '/' + group + '/' + group + '_'  +  str(int(w*10)) + '/'
                if not os.path.isdir(current_save):
                    os.makedirs(current_save, exist_ok=True)

                net_model3.to_excel(current_save + 'net_model_'  + group + str(int(w*10)) + '.xlsx')          
                        
                # create excel file in each subfolder 
                nn_file_name = 'net_model_' + group + str(int(w*10))
                net_model_file = os.path.join(current_save, nn_file_name + '.xlsx')
                    
                #: plots
                markers = ["r_hand_com"]
                coord_plot = ['r_shoulder_elev', 'r_elbow_flexion']
                musc_acti = "all"
                
                gen_net('arm', nn_file_name, net_model_file, step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=include,
                                save_folder=current_save, legend=True, show_plot=show_plot)

                #: Simulation
                save_kin = True 
                net_osim(osim_file, step_size, n_steps, nn_file_name, net_model_file, include, controls_dict, nn_init=nn_init, 
                        integ_acc=integ_acc, perturb=perturb, markers=markers, coord_plot=coord_plot, musc_acti=musc_acti, 
                        rates_plot=rates_plot, save_folder=current_save, visualize=visualize, show_plot=show_plot, save_kin=save_kin)
    
    elif len(fixed_pathways)==1:
        for group in groups:
            net_model = net.copy()
            for w in weights:
                net_model,include1,_, rates_plot = select_connections(fixed_pathways[0], net_model, fixed_weights[0], save_group=False)
                net_model2,include2,_, rates_plot = select_connections(group, net_model, w, save_group=False)
                include = include1 + include2 

                if not os.path.isdir (save_folder + 'fixed_' + fixed_pathways[0] + '/'):
                    os.makedirs( save_folder + 'fixed_' + fixed_pathways[0] + '/')
                
                if not os.path.isdir(save_folder + 'fixed_' + fixed_pathways[0] + '/' + group + '/'):
                    os.makedirs(save_folder + 'fixed_' + fixed_pathways[0] + '/' + group + '/')

                if not os.path.isdir(save_folder + 'fixed_' + fixed_pathways[0] + '/' + group + '/' + group + '_'  +  str(int(w*10)) + '/'):
                    os.makedirs(save_folder + 'fixed_' + fixed_pathways[0] + '/' + group + '/' + group + '_'  +  str(int(w*10)) + '/')

                current_save = save_folder + 'fixed_' + fixed_pathways[0] +  '/' + group + '/' + group + '_'  +  str(int(w*10)) + '/'
                if not os.path.isdir(current_save):
                    os.makedirs(current_save, exist_ok=True)

                net_model2.to_excel(current_save + 'net_model_'  + group + str(int(w*10)) + '.xlsx')          
                        
                # create excel file in each subfolder 
                nn_file_name = 'net_model_' + group + str(int(w*10))
                net_model_file = os.path.join(current_save, nn_file_name + '.xlsx')

                    
                #: plots
                markers = ["r_hand_com"]
                coord_plot = ['r_shoulder_elev', 'r_elbow_flexion']
                musc_acti = "all"
                
                gen_net('arm', nn_file_name, net_model_file, step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=include,
                                save_folder=current_save, legend=True, show_plot=show_plot)

                #: Simulation
                save_kin = True 
                net_osim(osim_file, step_size, n_steps, nn_file_name, net_model_file, include, controls_dict, nn_init=nn_init, 
                        integ_acc=integ_acc, perturb=perturb, markers=markers, coord_plot=coord_plot, musc_acti=musc_acti, 
                        rates_plot=rates_plot, save_folder=current_save, visualize=visualize, show_plot=show_plot, save_kin=save_kin)
    
    else:
        for group in groups:
            net_model = net.copy()
            for w in weights:
                net_model,include1,_, rates_plot = select_connections(group, net_model, w, save_group=False) 
                include = include1 

                if not os.path.isdir (save_folder + 'fixed_/'):
                    os.makedirs( save_folder + 'fixed_/')
            
                if not os.path.isdir(save_folder + 'fixed_/' + group + '/'):
                    os.makedirs(save_folder + 'fixed_/' + group + '/')

                if not os.path.isdir(save_folder + 'fixed_/' + group + '/' + group + '_'  +  str(int(w*10)) + '/'):
                    os.makedirs(save_folder + 'fixed_/' + group + '/' + group + '_'  +  str(int(w*10)) + '/')

                current_save = save_folder + 'fixed_/' + group + '/' + group + '_'  +  str(int(w*10)) + '/'
                if not os.path.isdir(current_save):
                    os.makedirs(current_save, exist_ok=True)

                net_model.to_excel(current_save + 'net_model_'  + group + str(int(w*10)) + '.xlsx')  
                            
                    
                # create excel file in each subfolder 
                nn_file_name = 'net_model_' + group + str(int(w*10))
                net_model_file = os.path.join(current_save, nn_file_name + '.xlsx')
                
                #: plots
                markers = ["r_hand_com"]
                coord_plot = ['r_shoulder_elev', 'r_elbow_flexion']
                musc_acti = "all"
            
                gen_net('arm', nn_file_name, net_model_file, step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=include,
                            save_folder=current_save, legend=True, show_plot=show_plot)

                #: Simulation
                save_kin = True 
                net_osim(osim_file, step_size, n_steps, nn_file_name, net_model_file, include, controls_dict, nn_init=nn_init, 
                        integ_acc=integ_acc, perturb=perturb, markers=markers, coord_plot=coord_plot, musc_acti=musc_acti, 
                        rates_plot=rates_plot, save_folder=current_save, visualize=visualize, show_plot=show_plot, save_kin=save_kin)


def read_optim_param(optim_folder):

    res_file = optim_folder + "_best_results_input.txt"
    res_file = open(res_file, "r")
    res = res_file.readlines()[-1].split(", ")
    flex_params = np.float_(res[2:6])
    ext_params = np.float_(res[6:])

    return flex_params, ext_params


if __name__ == '__main__':
        main()
        