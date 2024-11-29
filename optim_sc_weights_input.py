""" Optimisation of the SC weights for various scenarios """

from net_osim import *
from osim_model import *
from optim_algo import *
import pandas as pd
import numpy as np
import math 
import json
import brain_inputs
from run_net_sim_groups_input import read_optim_param


neuron_model = 'leaky'  
tau = 0.001 # leaky time constant
bias = -0.5 # center sigmoid to 0.5
D = 8 # sigmoid slope at bias
aff_delay = 30.0 # afferent delay in miliseconds

def main():  

    # set scenario
    traj = "circle"  # "flexion" or "circle"
    optimization_algos = ["cma"]  # ["pso", "cma", "mu,lambda", "de"]
    loss_type = "rmse"  # "rmse", "mae", "sdtw"
    w_joints = [1, 2]  # shoulder, elbow weights loss function
    baseline = 1  # number of baseline trial
    
    # target traj
    if traj == "flexion":
        target_traj = "flexion_slow_2" 
        sim_period = 2.8  # in sec
    elif traj == "circle":
        target_traj = "circles_mod_4" 
        sim_period = 1.3  # in sec
    target_folder = "data/UP002/" + target_traj.split("_")[0] + "/" + target_traj.split("_")[1]+ "/" +  target_traj.split("_")[2] + "/" 
    target_file = target_folder + target_traj.split("_")[0]+"_" + target_traj.split("_")[1]+"_"+target_traj.split("_")[2]+"_position.txt"

    # Brain input 
    save_folder = "results/"+traj+"/optimisation/SC_weights/"
    optim_folder = "results/"+traj+"/optimisation/brain_input_SC/cma/baseline_1_"+target_traj+"/" 
    params_flex, params_ext = read_optim_param(optim_folder)
    print("Brain input param: ", params_flex, params_ext) 

    # SC pathways
    variable_connexions = ['Ia_Mn','Ia_In']  # The synaptic weights we want to optimise
    fixed_connexions = ['IaIn_Mn'] # The synaptic weights we want to keep fixed (at 1)
    nodes = ['Ia', 'IaIn'] # The neuron we include in the network (except Mn that is always included)
    max_weight = 2 # the upper bound of the synaptic weights that we optimise

    # Environments 
    gravity_magnitudes = np.array([2.0]) # the factor multiplying the earth gravity we want to test, np.linspace(smallest factor, largest factor, number of values)
    gravity_directions = np.array([-45]) ## the direction of the gravity we want to test, 0 is vertical, 90 is horizontal (i.e. equivalent to being lying down)
    traj_name = 'baseline_'+str(baseline)+'_'+target_traj
    gravities = {traj_name : [[0, -9.81, 0]], 
                 'magnitude': remove_baseline(np.round(np.vstack((np.zeros(len(gravity_magnitudes)), -9.81*gravity_magnitudes, np.zeros(len(gravity_magnitudes)))).T, 2)),
                 'direction': remove_baseline(np.round(np.vstack((-9.81*np.sin(gravity_directions*np.pi/180), -9.81*np.cos(gravity_directions*np.pi/180), np.zeros(len(gravity_directions)))).T, 2))}

    ###############################################################

    for algo in optimization_algos:
        optimization_algo = algo
        # opensim model
        n_muscles = 7
        step_size = 0.01
        integ_acc = 0.0001
        # Simulation period
        n_steps = int(sim_period/step_size)    
        time_points = np.linspace(0, sim_period, n_steps)

        # SC net model
        nn_file_name = 'net'
        nn_init = 0  
        n_param = len(variable_connexions)*n_muscles # number of parameters to optimise

        # Brain input
        controls_dict_in = brain_inputs.brain_inputs(params_flex, params_ext, time_points)
        #for control in controls :   
        controls_dict = dict()
        controls_dict["Mn"] = dict()
        delay = 0  # delay before controls in sec
        delay = int(delay/step_size)
        if n_muscles > 2:
            controls_dict["Mn"]["BIClong"] = controls_dict_in["signal_delt"]
            controls_dict["Mn"]["BICshort"] = controls_dict_in["signal_values_flex"]
            controls_dict["Mn"]["BRA"] = controls_dict_in["signal_values_flex"]
            controls_dict["Mn"]["DELT1"] = controls_dict_in["signal_delt"]

            controls_dict["Mn"]["TRIlong"] = controls_dict_in["signal_values_ext"]
            controls_dict["Mn"]["TRIlat"] = controls_dict_in["signal_values_ext"]
            controls_dict["Mn"]["TRImed"] = controls_dict_in["signal_values_ext"]

        #: Perturbations
        perturb = None 

        # Net model parameters
        columns         = ['Muscles','type','Ia_antagonist','Ia_delay','Ia_Mn_w','Ia_In_w','IaIn_Mn_w','IaIn_IaIn_w','Mn_Rn_w',
                        'Rn_Mn_w','Rn_IaIn_w','Rn_Rn_w','II_delay','II_w','Ib_delay','Ib_w','IbIn_w']
        muscle_list     = ['DELT1', 'TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA']
        type_list       = ['elb_flex','elb_ext','elb_ext','elb_ext','elb_flex','elb_flex','elb_flex']
        antagonist_list = ['TRIlong','DELT1,BIClong','BICshort,BRA','BICshort,BRA','TRIlong','TRIlat,TRImed','TRIlat,TRImed']
        null_column = np.zeros(n_muscles)
        net = pd.DataFrame(zip(muscle_list,type_list,antagonist_list,null_column,null_column,null_column,null_column,
                            null_column,null_column,null_column,null_column,null_column,null_column,null_column,
                            null_column,null_column,null_column), columns=columns)
        
        # Optimization algorithm parameters
        max_evals = 2002
        pop_size = 11
        params_algo = {
            "max_weight": max_weight,
            "min_weight": 0,
        }
        if optimization_algo == 'cma':
            params_algo['pop_size'] = pop_size
            params_algo['max_iter'] = max_evals//params_algo['pop_size']

        if optimization_algo == 'mu,lambda':
            params_algo['pop_size'] = pop_size
            params_algo['max_iter'] = max_evals//params_algo['pop_size']
            params_algo['cxpb'] = 0.7
            params_algo['mutpb'] = 0.3
        if optimization_algo == 'pso':
            params_algo['pop_size'] = pop_size
            params_algo['max_iter'] = max_evals//params_algo['pop_size']
        if optimization_algo == 'de':
            params_algo['pop_size'] = pop_size
            params_algo['max_iter'] = max_evals//params_algo['pop_size']
            params_algo['CR'] = 0.25
            params_algo['F'] = 1.0

        #: Results folder
        if not os.path.isdir("results/"):
            os.mkdir("results/")
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        if not os.path.isdir(save_folder + algo + '/'):   ######################
            os.mkdir(save_folder + algo + '/')

        
        # Loop over different environments
        for scenario in gravities:
            for gravity in gravities[scenario]:        
                current_folder = save_folder + algo + '/' + scenario
                if 'baseline' in scenario:
                    current_folder += "/"
                elif scenario == 'magnitude':
                    current_folder += '_x' + str(np.round(-gravity[1]/9.81,1)) + "/"
                elif scenario == 'direction':
                    current_folder += '_' + str(round(math.degrees(math.atan2(gravity[0], gravity[1])) +180 % 360)) + "/"
                else :
                    print("Error: scenario not recognised")
                    break
                if not os.path.isdir(current_folder):
                    os.mkdir(current_folder)

                # Store optimisation and model informations
                optim_infos(current_folder, optimization_algo, params_algo, loss_type, gravity, perturb, fixed_connexions, variable_connexions, 
                            nodes, n_param, n_muscles, step_size, integ_acc, n_steps, sim_period, controls_dict, delay, target_file)
                
                # Run optimisation
                optim = Multi_Optim(net, target_file, loss_type, optimization_algo, params_algo, w_joints, current_folder, n_muscles, step_size, integ_acc, n_steps, sim_period, nn_file_name, nodes, fixed_connexions, variable_connexions, nn_init, controls_dict, perturb, gravity)
                optim.run_optim()
            

def remove_baseline(arrays, baseline = [0, -9.81, 0]):

    mask = [np.allclose(a, baseline) for a in arrays]
    return [a for a, skip in zip(arrays, mask) if not skip]


def optim_infos(current_folder, optimization_algo, algo_params, loss_type, gravity, perturb, fixed_connexions, variable_connexions, 
                      nodes, n_param, n_muscles, step_size, integ_acc, n_steps, sim_period, controls_dict, delay,target_traj) :
    # Store optimisation and model informations and parameters in list containing each info as a list item
    optim_infos = ["Optimisation algorithm: " + optimization_algo + '\n', "Algorithm parameter: " + json.dumps(algo_params) + '\n', 
                   "Loss type: " + loss_type + '\n', "Gravity: " + str(gravity) + '\n', "Perturbation: " + str(perturb) + '\n',
                   "Fixed connexions: " + str(fixed_connexions) + '\n', "Variable connexions: " + str(variable_connexions) + '\n',
                   "Nodes: " + str(nodes) + '\n', "Number of parameters: " + str(n_param) + '\n', "Number of muscles: " + str(n_muscles) + '\n',
                   "Step size: " + str(step_size) + '\n', "Integration accuracy: " + str(integ_acc) + '\n', "Number of steps: " + str(n_steps) + '\n',
                   "Simulation period: " + str(sim_period) + '\n', "Input amplitude: " + str(controls_dict) + '\n', "Input delay: " + str(delay*step_size) + '\n',
                    "Target trajectory: " + str(target_traj) + '\n']
    
    # If there is already a checkpoint file, check if the optimisation parameters are the same (else delete the checkpoint file)
    if os.path.isfile(current_folder + 'checkpoint.pkl'):
        with open(current_folder + 'optim_info.txt', 'r') as f:
            cp_infos = f.readlines()
        if cp_infos != optim_infos:
            os.remove(current_folder + 'checkpoint.pkl')
            os.remove(current_folder + 'best_results.txt')
            print(f"previous run was done with different parameters, so the checkpoint files was deleted")

    # Store current optimisation and model informations and parameters in a file
    with open(current_folder + 'optim_info.txt', 'w') as f:
        f.writelines(optim_infos)


class Multi_Optim:
    
    def __init__(self, net, target_file, loss_type, optim_algo, params_algo, w_joints, save_folder, n_muscles, step_size, integ_acc, n_steps, sim_period,
                 nn_file_name, nodes, fixed_connexions, variable_connexions, nn_init, controls_dict, perturb, gravity=None):

        # optim param
        self.net = net
        self.target_file = target_file
        self.loss_type = loss_type
        self.save_folder = save_folder
        self.w_joints = w_joints
        self.optim_algo = optim_algo
        self.params_algo = params_algo

        # opensim model
        if n_muscles == 7:
            self.osim_file = 'models/full_arm_M_delt1_wrap.osim'
            self.osim_file = lock_Coord(self.osim_file, ['r_shoulder_elev'], 'false')  # "true" if you want to lock the shoulder
            self.osim_file = lock_Coord(self.osim_file, ['elv_angle', 'shoulder_rot'], 'true')  # "fasle" if you want to unlock the shoulder in 3D
        elif n_muscles == 8:
            self.osim_file = 'models/full_arm_M_delt1and3_wrap.osim'
            self.osim_file = lock_Coord(self.osim_file, ['r_shoulder_elev'], 'false')  # "true" if you want to lock the shoulder
            self.osim_file = lock_Coord(self.osim_file, ['elv_angle', 'shoulder_rot'], 'true')  # "fasle" if you want to unlock the shoulder in 3D
        self.step_size = step_size
        self.integ_acc = integ_acc
        self.n_steps = n_steps
        self.sim_period = sim_period

        # SC net model
        self.nn_file_name = nn_file_name
        self.nodes = nodes
        self.fixed_connexions = fixed_connexions
        self.variable_connexions = variable_connexions
        self.include = self.nodes + self.variable_connexions + self.fixed_connexions
        self.nn_init = nn_init
        self.n_muscles = n_muscles

        # initialise delays
        for node in self.nodes :
            self.net[node + '_delay'] = aff_delay*np.ones(self.n_muscles)
        for connexion in self.fixed_connexions :
            self.net[connexion + '_w'] = np.ones(self.n_muscles)
        
        # scenario
        self.controls_dict = controls_dict
        self.perturb = perturb
        self.gravity = gravity
        self.visualize = False
        self.show_plot = True
        self.save_kin = True

        #: plots
        self.markers = ["r_hand_com"]
        if n_muscles == 2:
            self.coord_plot = ['r_elbow_flexion']
        else:
            self.coord_plot = ['r_shoulder_elev', 'r_elbow_flexion']
        self.musc_acti = "all"

        #: rates plot
        if bool(self.include):
            # all afferents test
            all_afferents = ['Ia', 'IaIn', 'IaIn_IaIn', 'Rn', 'Rn_IaIn', 'Rn_Rn', 'II', 'Ib', 'IbIn', 'IbIn_Mn']
            if len(self.include) == len(all_afferents):
                self.rates_plot = ['all', ['Mn', self.include[0], self.include[1], self.include[3], self.include[6], self.include[7], self.include[8]]]
            # just one afferent test
            elif (self.include[0] == 'Ia') or (self.include[0] == 'Ib'):
                self.rates_plot = ['all', ['Mn', self.include[0], self.include[1]]] # [[muscles], [neurons]]
            elif (self.include[0]=='II') or (self.include[0]=='Rn'):                                                     #: TO DO : change II or Rn if In connections added !
                self.rates_plot = ['all', ['Mn', self.include[0]]] # [[muscles], [neurons]]
        # none afferent test
        else:
            self.rates_plot = ['all', ['Mn']]

    def run_optim(self):
        # Create savefile
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        n_param = len(self.variable_connexions)*self.n_muscles

        # pack parameters for optimisation
        args = {"loss" : self.multi_loss, "n_weights" : n_param, "output_folder" : self.save_folder, "final_run" : self.best_net_run, **self.params_algo}

        # Choose optimisation algorithm
        if self.optim_algo == 'cma':
            optim_algo = CMA(**args)
        elif self.optim_algo == 'pso':
            optim_algo = PSO(**args)
        elif self.optim_algo == 'mu,lambda':
            optim_algo = MuCommaLambda(**args)
        elif self.optim_algo == 'de':
            optim_algo = DE(**args)
        else:
            raise ValueError("Algorithm not recognised")
                
        # Run optimisation
        optim_algo.run()


    def multi_loss(self, param):

        loss = self.trial_fun_joint(param=param)

        return [loss]


    def trial_fun_joint(self, param):

        # update net
        for i, connexion in enumerate(self.variable_connexions) :
            self.net[connexion + '_w'] = param[i*self.n_muscles:(i+1)*self.n_muscles]

        net_model_file = self.save_folder + 'net_model.xlsx'
        self.net.to_excel(net_model_file)

        gen_net('arm', self.nn_file_name, net_model_file, self.step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=self.include,
        save_folder=self.save_folder, legend=True, show_plot=False)

        # run simulation
        sh_pos, elb_pos = net_osim(osim_file=self.osim_file, step_size=self.step_size, n_steps=self.n_steps, nn_file_name=self.nn_file_name, net_model_file=net_model_file, 
                                   include=self.include, controls_dict=self.controls_dict, nn_init=self.nn_init, integ_acc=self.integ_acc,  save_folder=self.save_folder,
                                   perturb=self.perturb, gravity=self.gravity, coord_plot=self.coord_plot, musc_acti=self.musc_acti, 
        visualize = False, show_plot = False, save_kin = True)

        # target trajectory 
        target = pd.read_csv(self.target_file, sep=' ')
        sh_target = target.iloc[:, 0].values
        elb_target = target.iloc[:, 1].values
        sh_target0 = target.iloc[:, 0].values[0]
        elb_target0 = target.iloc[:, 1].values[0]
        modify_default_Coord(self.osim_file, 'r_shoulder_elev', sh_target0)
        modify_default_Coord(self.osim_file, 'r_elbow_flexion', elb_target0)

        # resample/interpolate ref angle so that it has the same length as the simulated angle
        sh_target = np.interp(np.linspace(0, self.sim_period, len(sh_pos)), np.linspace(0, self.sim_period, len(sh_target)), sh_target)
        elb_target = np.interp(np.linspace(0, self.sim_period, len(elb_pos)), np.linspace(0, self.sim_period, len(elb_target)), elb_target)

        """if self.loss_type == "sdtw":
            D_sh = SquaredEuclidean(sh_pos.reshape(-1, 1), sh_target.reshape(-1, 1))
            sdtw_hip = SoftDTW(D_sh, gamma=1.0)
            # soft-DTW discrepancy, approaches DTW as gamma -> 0
            value_sh = sdtw_hip.compute()

            D_elb = SquaredEuclidean(elb_pos.reshape(-1, 1), elb_target.reshape(-1, 1))
            sdtw_knee = SoftDTW(D_elb, gamma=1.0)
            # soft-DTW discrepancy, approaches DTW as gamma -> 0
            value_elb = sdtw_knee.compute()

            loss = self.w_joints[0]*value_sh + self.w_joints[1]*value_elb"""
        
        if self.loss_type == "rmse":

            loss = self.w_joints[0]*np.sqrt(np.mean(np.square(sh_pos-sh_target))) + self.w_joints[1]*np.sqrt(np.mean(np.square(elb_pos-elb_target)))

        elif self.loss_type == "mae":

            loss = self.w_joints[0]*np.mean(np.abs(sh_pos-sh_target)) + self.w_joints[1]*np.mean(np.abs(elb_pos-elb_target))

        return loss
    
    def best_net_run(self, param):
        # update net
        for i, connexion in enumerate(self.variable_connexions) :
            self.net[connexion + '_w'] = param[i*self.n_muscles:(i+1)*self.n_muscles]

        net_model_file = self.save_folder + 'net_model.xlsx'
        self.net.to_excel(net_model_file)

        gen_net('arm', self.nn_file_name, net_model_file, self.step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=self.include,
        save_folder=self.save_folder, legend=True, show_plot=False)

        # run simulation
        net_osim(osim_file=self.osim_file, step_size=self.step_size, n_steps=self.n_steps, nn_file_name=self.nn_file_name, net_model_file=net_model_file, include=self.include,
                 controls_dict=self.controls_dict, nn_init=self.nn_init, integ_acc=self.integ_acc, save_folder=self.save_folder, perturb=self.perturb, gravity=self.gravity, markers=None, 
                 coord_plot=self.coord_plot, musc_acti=self.musc_acti, rates_plot=None, visualize=False, 
                 show_plot=False, save_kin=True)

if __name__ == '__main__':
    main()
