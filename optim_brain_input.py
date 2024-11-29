""" Optimisation of the brain inputs for various scenarios """

from dataset import *
from net_osim import *
from osim_model import *
from optim_algo_input import *
import pandas as pd
import numpy as np
import math
import json
import brain_inputs


# neuron parameters
neuron_model = 'leaky'  
tau = 0.001 # leaky time constant
bias = -0.5 # center sigmoid to 0.5
D = 8 # sigmoid slope at bias
aff_delay = 30.0 # afferent delay in miliseconds

def main():    

    # set scenario
    traj = "circle"  # "flexion" or "circle"
    with_sc = True  # True to optimise with minimal SC    
    baseline = 1  # number of baseline trial
    optimization_algo = "cma"  # "cma", "mu,lambda", "pso", "de"
    loss_type = "rmse"  # "sdtw", "mae", "sdtw"
    w_joints = [1, 2]  # shoulder, elbow weights loss function

    # target traj
    save_folder = "results/" + traj + "/optimisation/brain_input" 
    if traj == "flexion":
        target_traj = "flexion_slow_2" 
        sim_period = 2.8  # in sec
    elif traj == "circle":
        target_traj = "circles_mod_4" 
        sim_period = 1.3  # in sec
    target_folder = "data/UP002/" + target_traj.split("_")[0] + "/" + target_traj.split("_")[1]+ "/" +  target_traj.split("_")[2] + "/" 
    target_file = target_folder + target_traj.split("_")[0]+"_" + target_traj.split("_")[1]+"_"+target_traj.split("_")[2]+"_position.txt"

    # SC pathways
    if not with_sc:
        fixed_connexions = [] 
        nodes = [] 
        fixed_weights = [1, 1, 1]
    else:
        save_folder += "_SC"
        fixed_connexions = ['Ia_Mn', "IaIn_Mn", 'Ia_In'] #['Ia_Mn', "IaIn_Mn", 'Ia_In'] #['Ia_Mn', 'Ia_Mn_syn',"IaIn_Mn", "IaIn_IaIn", 'Ia_In','Ib', 'IbIn','Ib_In','II', 'Mn_Rn','Rn_Mn', 'Rn_IaIn', 'Rn_Rn'] # The synaptic weights we want to keep fixed (at 1)
        nodes = ['Ia', "IaIn"] #['Ia', "IaIn"] # ['Ia', "IaIn", "Rn", "II", "Ib",'Mn_syn', 'IbIn'] The neuron we include in the network (except Mn that is always included)
        fixed_weights = [1, 1, 1]

    # Environments 
    gravity_magnitudes = np.array([0.5, 1.5, 2]) # the factor multiplying the earth gravity we want to test, np.linspace(smallest factor, largest factor, number of values)
    gravity_directions = np.array([45, 90, -45]) # the direction of the gravity we want to test, 0 is vertical, 90 is horizontal (i.e. equivalent to being lying down)
    traj_name = 'baseline_'+str(baseline)+'_'+target_traj
    gravities = {traj_name : [[0, -9.81, 0]],
                 'magnitude': remove_baseline(np.round(np.vstack((np.zeros(len(gravity_magnitudes)), -9.81*gravity_magnitudes, np.zeros(len(gravity_magnitudes)))).T, 2)),
                 'direction': remove_baseline(np.round(np.vstack((-9.81*np.sin(gravity_directions*np.pi/180), -9.81*np.cos(gravity_directions*np.pi/180), np.zeros(len(gravity_directions)))).T, 2))}
                    
    # save folder
    if not os.path.isdir("results/"):
        os.mkdir("results/")
    if not os.path.isdir("results/" + traj):
        os.mkdir("results/" + traj)
    if not os.path.isdir("results/" + traj + "/optimisation"):
        os.mkdir("results/" + traj + "/optimisation")

    ###############################################################

    #for algo in ["cma", "pso", "mu,lambda", "de"]:
    for algo in ["cma"]:
        optimization_algo = algo
        # opensim model
        n_muscles = 7
        step_size = 0.01
        integ_acc = 0.0001
        n_steps = int(sim_period/step_size)

        # SC net mode
        nn_file_name = 'net'
        nn_init = 0  
        #max_weight = 2 # the upper bound of the synaptic weights that we optimise
        n_param = 8 # number of parameters to optimise (offset, amplitude, frequency and phase for ext and flex)

        #: Descending controls - flexion example
        controls_dict = dict()
        controls_dict["Mn"] = dict()
        delay = 0  # delay before controls in sec
        delay = int(delay/step_size)
       
        def generate_random_params(bounds):
            return random.uniform(bounds[0], bounds[1])

        # Bounds for amplitude, frequency, and phase for flex and ext signals
        offset_bounds_flex = [0, 0.1]
        amplitude_bounds_flex = [0.05, 0.7]  # Adjust the bounds as needed
        frequency_bounds_flex = [0.1, 0.8]  # Adjust the bounds as needed
        phase_bounds_flex = [0.0, 2 * np.pi]  # Adjust the bounds as needed

        offset_bounds_ext = [0, 0.1]
        amplitude_bounds_ext = [0.05, 0.7]  # Adjust the bounds as needed
        frequency_bounds_ext = [0.1, 0.8]  # Adjust the bounds as needed
        phase_bounds_ext = [0.0, 2 * np.pi]  # Adjust the bounds as needed

        # Generate random values within bounds for params_flex
        params_flex = [
            generate_random_params(offset_bounds_flex),
            generate_random_params(frequency_bounds_flex),
            generate_random_params(amplitude_bounds_flex),
            generate_random_params(phase_bounds_flex)
        ]

        # Generate random values within bounds for params_ext
        params_ext = [
            generate_random_params(offset_bounds_ext),
            generate_random_params(frequency_bounds_ext),
            generate_random_params(amplitude_bounds_ext),
            generate_random_params(phase_bounds_ext)
        ]
        # Generate time points for the signals
        time_points = np.linspace(0, sim_period, n_steps)

        # Call the brain_input function with the generated parameters
        controls_dict_in = brain_inputs.brain_inputs(params_flex, params_ext, time_points)


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
        max_evals = 2750

        pop_size = 11
        params_algo = {
            "max_bounds": [offset_bounds_flex[1], frequency_bounds_flex[1],amplitude_bounds_flex[1],phase_bounds_flex[1],offset_bounds_ext[1],frequency_bounds_ext[1],amplitude_bounds_ext[1],phase_bounds_ext[1]], #[f_max_flex, a_max_flex, phase_max_flex,f_max_ext, a_max_ext, phase_max_ext]
            "min_bounds": [offset_bounds_flex[0], frequency_bounds_flex[0],amplitude_bounds_flex[0],phase_bounds_flex[0],offset_bounds_ext[0],frequency_bounds_ext[0],amplitude_bounds_ext[0],phase_bounds_ext[0]], #[f_min_flex, a_min_flex, phase_min_flex,f_min_ext, a_min_ext, phase_min_ext]
        }
        params_algo_int={}
        if optimization_algo == 'cma':
            params_algo_int['pop_size'] = pop_size
            params_algo_int['max_iter'] = max_evals//params_algo_int['pop_size']

        if optimization_algo == 'mu,lambda':
            params_algo_int['pop_size'] = pop_size
            params_algo_int['max_iter'] = max_evals//params_algo_int['pop_size']
            params_algo_int['cxpb'] = 0.7
            params_algo_int['mutpb'] = 0.3
        if optimization_algo == 'pso':
            params_algo_int['pop_size'] = pop_size
            params_algo_int['max_iter'] = max_evals//params_algo_int['pop_size']
        if optimization_algo == 'de':
            params_algo_int['pop_size'] = pop_size
            params_algo_int['max_iter'] = max_evals//params_algo_int['pop_size']
            params_algo_int['CR'] = 0.25
            params_algo_int['F'] = 1.0
        #print(params_algo)

        #: Results folder
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        if not os.path.isdir(save_folder + "/" + algo + '/'):  
            os.mkdir(save_folder + "/" + algo + '/')

        
        # Loop over different environments
        for scenario in gravities:
            for gravity in gravities[scenario]:        
                current_folder = save_folder + "/" + algo + '/' + scenario
                if 'baseline' in scenario:
                    current_folder = current_folder + '/'
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
                optim_infos(current_folder, optimization_algo, params_algo, params_algo_int, loss_type, gravity, perturb, fixed_connexions, fixed_weights,
                            nodes, n_param, n_muscles, step_size, integ_acc, n_steps=n_steps, sim_period=sim_period, control=controls_dict, delay=delay, params_flex=params_flex, params_ext=params_ext, target_traj=target_file)
                
                # Run optimisation
                optim = Multi_Optim(net, target_file, loss_type, optimization_algo, params_algo, params_algo_int, w_joints, current_folder, n_muscles, step_size, integ_acc, n_steps, sim_period, nn_file_name, nodes, 
                                    fixed_connexions, fixed_weights, nn_init, controls_dict, perturb, params_flex, params_ext, gravity)
                #print("dict:", params_algo_int["pop_size"])
                optim.run_optim()                
            

def remove_baseline(arrays, baseline = [0, -9.81, 0]):

    mask = [np.allclose(a, baseline) for a in arrays]
    return [a for a, skip in zip(arrays, mask) if not skip]


def optim_infos(current_folder, optimization_algo, algo_params, params_algo_int, loss_type, gravity, perturb, fixed_connexions, fixed_weights,
                      nodes, n_param, n_muscles, step_size, integ_acc, n_steps, sim_period, control, delay, params_flex, params_ext, target_traj) :
    # Store optimisation and model informations and parameters in list containing each info as a list item
    optim_infos = ["Optimisation algorithm: " + optimization_algo + '\n', "Algorithm parameters to opt: " + json.dumps(algo_params) + '\n', 
                    "Algorithm parameters: " + json.dumps(params_algo_int) + '\n',"Loss type: " + loss_type + '\n', "Gravity: " + str(gravity) + '\n', 
                    "Perturbation: " + str(perturb) + '\n', "Fixed connexions: " + str(fixed_connexions)  + '\n', "Fixed weights: " + str(fixed_weights) + '\n', "Nodes: " + str(nodes) + '\n', "Number of parameters: " + str(n_param) + '\n', 
                   "Number of muscles: " + str(n_muscles) + '\n',"Step size: " + str(step_size) + '\n', "Integration accuracy: " + str(integ_acc) + '\n', 
                   "Number of steps: " + str(n_steps) + '\n',
                   "Simulation period: " + str(sim_period) + '\n', "Input amplitude: " + str(control) + '\n', "Input delay: " + str(delay*step_size) + '\n',
                   "Flexion Parameters: "+ str(params_flex)+'\n', "Extension Parameters: "+ str(params_ext)+'\n',
                    "Target trajectory: " + str(target_traj) + '\n',]
    
    # If there is already a checkpoint file, check if the optimisation parameters are the same (else delete the checkpoint file)
    if os.path.isfile(current_folder + '_checkpoint_input.pkl'):
        with open(current_folder + 'input_optim_info.txt', 'r') as f:
            cp_infos = f.readlines()
        if cp_infos != optim_infos:

            os.remove(current_folder + 'checkpoint_input.pkl')
            os.remove(current_folder + 'best_results_input.txt')
            print(f"previous run was done with different parameters, so the checkpoint files was deleted")

    # Store current optimisation and model informations and parameters in a file
    with open(current_folder + 'input_optim_info.txt', 'w') as f:
        f.writelines(optim_infos)


class Multi_Optim:
    
    def __init__(self, net, target_file, loss_type, optim_algo_input, params_algo, params_algo_int, w_joints, save_folder, n_muscles, step_size, integ_acc, n_steps, sim_period,
                 nn_file_name, nodes, fixed_connexions, fixed_weights, nn_init, controls_dict, perturb, params_flex, params_ext, gravity=None):

        # optim param
        self.net = net
        self.target_file = target_file
        self.loss_type = loss_type
        self.save_folder = save_folder
        self.w_joints = w_joints
        self.optim_algo_input = optim_algo_input
        self.params_algo = params_algo
        self.params_flex=params_flex
        self.params_ext=params_ext
        self.time_points=np.linspace(0, sim_period, n_steps)
        self.params_algo_int=params_algo_int

        # opensim model
        if n_muscles == 7:
            self.osim_file = 'models/full_arm_M_delt1_wrap.osim'
            self.osim_file = lock_Coord(self.osim_file, ['r_shoulder_elev'], 'false')  # "true" if you want to lock the shoulder
            self.osim_file = lock_Coord(self.osim_file, ['elv_angle', 'shoulder_rot'], 'true')  # "false" if you want to unlock the shoulder in 3D
        elif n_muscles == 8:
            self.osim_file = 'models/full_arm_M_delt1and3_wrap.osim'
            self.osim_file = lock_Coord(self.osim_file, ['r_shoulder_elev'], 'false')  # "true" if you want to lock the shoulder
            self.osim_file = lock_Coord(self.osim_file, ['elv_angle', 'shoulder_rot'], 'true')  # "false" if you want to unlock the shoulder in 3D
        self.step_size = step_size
        self.integ_acc = integ_acc
        self.n_steps = n_steps
        self.sim_period = sim_period

        # init position osim model
        target = pd.read_csv(self.target_file, sep=' ')
        sh_target0 = target.iloc[:, 0].values[0]
        elb_target0 = target.iloc[:, 1].values[0]
        modify_default_Coord(self.osim_file, 'r_shoulder_elev', sh_target0)
        modify_default_Coord(self.osim_file, 'r_elbow_flexion', elb_target0)

        # SC net model
        self.nn_file_name = nn_file_name
        self.nodes = nodes
        self.fixed_connexions = fixed_connexions
        self.include = self.nodes + self.fixed_connexions
        self.fixed_weights = fixed_weights
        self.nn_init = nn_init
        self.n_muscles = n_muscles

        # initialise delays
        for node in self.nodes :
            self.net[node + '_delay'] = aff_delay*np.ones(self.n_muscles)
        for c, connexion in enumerate(self.fixed_connexions):
            self.net[connexion + '_w'] = self.fixed_weights[c]*np.ones(self.n_muscles)
        
        # scenario
        self.controls_dict = controls_dict
        self.perturb = perturb
        self.gravity = gravity
        self.visualize =False
        self.show_plot = False
        self.save_kin = True

        #: plots
        self.markers = ["r_hand_com"]
        if n_muscles == 2:
            self.coord_plot = ['r_elbow_flexion']
        else:
            self.coord_plot = ['r_shoulder_elev', 'r_elbow_flexion']  ## DO NOT CHANGE
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
        #n_param = len(self.fixed_connexions)*self.n_muscles
        n_param = 8
        max_bounds=self.params_algo["max_bounds"]
        min_bounds=self.params_algo["min_bounds"]


        # pack parameters for optimisation
        args = {"loss" : self.multi_loss, "n_params" : n_param, "max_bounds":max_bounds, "min_bounds":min_bounds, "output_folder" : self.save_folder, **self.params_algo_int, "final_run" : self.best_net_run,}

        # Choose optimisation algorithm
        if self.optim_algo_input == 'cma':
            #print('args: ', args)
            optim_algo_input = CMA(**args)
        elif self.optim_algo_input == 'pso':
            optim_algo_input = PSO(**args)
        elif self.optim_algo_input == 'mu,lambda':
            optim_algo_input = MuCommaLambda(**args)
        elif self.optim_algo_input == 'de':
            optim_algo_input = DE(**args)
        else:
            raise ValueError("Algorithm not recognised")
                
        # Run optimisation
        optim_algo_input.run()


    def multi_loss(self, param):

        loss = self.trial_fun_joint(param=param)

        return [loss]


    def trial_fun_joint(self, param):
        
        offset_flex, frequency_flex, amplitude_flex, phase_flex=param[:4]
        offset_ext, frequency_ext, amplitude_ext, phase_ext=param[4:8]

        #print ("param:",param)
        
        self.params_flex=[offset_flex, frequency_flex, amplitude_flex, phase_flex]
        self.params_ext=[offset_ext, frequency_ext, amplitude_ext, phase_ext]
        
        # net
        for c, connexion in enumerate(self.fixed_connexions) :
            self.net[connexion] =  self.fixed_weights[c]

        net_model_file = self.save_folder + 'input_net_model.xlsx'
        self.net.to_excel(net_model_file)

        gen_net('arm', self.nn_file_name, net_model_file, self.step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=self.include,
        save_folder=self.save_folder, legend=True, show_plot=False)

        # update control dict 
        controls_dict_in = brain_inputs.brain_inputs(self.params_flex, self.params_ext, self.time_points)
        #print(controls_dict_in)
        self.controls_dict["Mn"]["BIClong"] = controls_dict_in["signal_delt"]
        self.controls_dict["Mn"]["BICshort"] = controls_dict_in["signal_values_flex"]
        self.controls_dict["Mn"]["BRA"] = controls_dict_in["signal_values_flex"]
        self.controls_dict["Mn"]["DELT1"] = controls_dict_in["signal_delt"]

        self.controls_dict["Mn"]["TRIlong"] = controls_dict_in["signal_values_ext"]
        self.controls_dict["Mn"]["TRIlat"] = controls_dict_in["signal_values_ext"]
        self.controls_dict["Mn"]["TRImed"] = controls_dict_in["signal_values_ext"]

        """plt.figure()
        muscle_list     = ['DELT1', 'TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA']
        for m in muscle_list:
            plt.plot(self.controls_dict["Mn"][m], label=m)
            plt.title(str(param))
        plt.legend()
        plt.show()"""
        
        # run simulation
        #print(self.controls_dict)
        sh_pos, elb_pos = net_osim(self.osim_file, self.step_size, self.n_steps, self.nn_file_name, net_model_file, self.include,
                                         self.controls_dict, self.save_folder, self.nn_init, self.integ_acc, self.perturb, self.gravity, self.markers, self.coord_plot, self.musc_acti,
                                         visualize = False, show_plot = False, save_kin = True)
        
        # target trajectory 
        target = pd.read_csv(self.target_file, sep=' ')
        sh_target = target.iloc[:, 0].values
        elb_target = target.iloc[:, 1].values
        
        # resample/interpolate ref angle so that it has the same length as the simulated angle
        sh_target = np.interp(np.linspace(0, self.sim_period, len(sh_pos)), np.linspace(0, self.sim_period, len(sh_target)), sh_target)
        elb_target = np.interp(np.linspace(0, self.sim_period, len(elb_pos)), np.linspace(0, self.sim_period, len(elb_target)), elb_target)
        
        #if self.loss_type == "sdtw":
        #    D_sh = SquaredEuclidean(sh_pos.reshape(-1, 1), sh_target.reshape(-1, 1))
        #    sdtw_hip = SoftDTW(D_sh, gamma=1.0)
            # soft-DTW discrepancy, approaches DTW as gamma -> 0
        #    value_sh = sdtw_hip.compute()

         #   D_elb = SquaredEuclidean(elb_pos.reshape(-1, 1), elb_target.reshape(-1, 1))
         #   sdtw_knee = SoftDTW(D_elb, gamma=1.0)
            # soft-DTW discrepancy, approaches DTW as gamma -> 0
         #   value_elb = sdtw_knee.compute()

         #   loss = self.w_joints[0]*value_sh + self.w_joints[1]*value_elb
        
        if self.loss_type == "rmse":

            loss_nopenalty = self.w_joints[0]*np.sqrt(np.mean(np.square(sh_pos-sh_target))) + self.w_joints[1]*np.sqrt(np.mean(np.square(elb_pos-elb_target)))
            control_penalty=0
            counter=0
            penalty_weight=5
            for i in range(len(self.time_points)):
                if self.time_points[i]<=0.1:
                    counter+=1
                    control_penalty+=(np.square(self.controls_dict["Mn"]["BIClong"][i])+np.square(self.controls_dict["Mn"]["BICshort"][i])+ np.square(self.controls_dict["Mn"]["BRA"][i])+np.square(self.controls_dict["Mn"]["DELT1"][i])+np.square(self.controls_dict["Mn"]["TRIlong"][i])+np.square(self.controls_dict["Mn"]["TRIlat"][i])+ np.square(self.controls_dict["Mn"]["TRImed"][i]))/self.n_muscles
                elif self.time_points[i]>=2.7:
                    counter+=1
                    control_penalty+=(np.square(self.controls_dict["Mn"]["BIClong"][i])+np.square(self.controls_dict["Mn"]["BICshort"][i])+ np.square(self.controls_dict["Mn"]["BRA"][i])+np.square(self.controls_dict["Mn"]["DELT1"][i])+np.square(self.controls_dict["Mn"]["TRIlong"][i])+np.square(self.controls_dict["Mn"]["TRIlat"][i])+ np.square(self.controls_dict["Mn"]["TRImed"][i]))/self.n_muscles
                else:
                    control_penalty+=0
            loss= loss_nopenalty+penalty_weight*control_penalty/counter

            #print("loss no penalty: ",loss_nopenalty)
            #print("counter: ",counter)
            #print("control penalty: ",control_penalty/counter)
            #print("loss: ",loss)
            #print("controls:",  self.controls_dict["Mn"])

            """plt.figure()
            plt.plot(elb_pos, label="elb sim")
            plt.plot(elb_target, label="elb target")
            plt.plot(sh_pos, label="sh sim")
            plt.plot(sh_target, label="sh target")
            plt.title(str(loss)+" "+str(param))
            plt.legend()
            plt.show()"""

        elif self.loss_type == "mae":

            loss = self.w_joints[0]*np.mean(np.abs(sh_pos-sh_target)) + self.w_joints[1]*np.mean(np.abs(elb_pos-elb_target))

        return loss
    
    
    def best_net_run(self, param):

        offset_flex, frequency_flex, amplitude_flex, phase_flex=param[:4]
        offset_ext, frequency_ext, amplitude_ext, phase_ext=param[4:8]

        
        self.params_flex=[offset_flex, frequency_flex, amplitude_flex, phase_flex]
        self.params_ext=[offset_ext, frequency_ext, amplitude_ext, phase_ext]
        

        for c, connexion in enumerate(self.fixed_connexions) :
            self.net[connexion] =  self.fixed_weights[c]
        
        net_model_file = self.save_folder + 'input_net_model.xlsx'
        self.net.to_excel(net_model_file)
        
        gen_net('arm', self.nn_file_name, net_model_file, self.step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=self.include,
        save_folder=self.save_folder, legend=True, show_plot=False)

        # Best control dict 
        controls_dict_in = brain_inputs.brain_inputs(self.params_flex, self.params_ext, self.time_points)
        print("time: ", self.time_points)

        self.controls_dict["Mn"]["BIClong"] = controls_dict_in["signal_delt"]
        self.controls_dict["Mn"]["BICshort"] = controls_dict_in["signal_values_flex"]
        self.controls_dict["Mn"]["BRA"] = controls_dict_in["signal_values_flex"]
        self.controls_dict["Mn"]["DELT1"] = controls_dict_in["signal_delt"]

        self.controls_dict["Mn"]["TRIlong"] = controls_dict_in["signal_values_ext"]
        self.controls_dict["Mn"]["TRIlat"] = controls_dict_in["signal_values_ext"]
        self.controls_dict["Mn"]["TRImed"] = controls_dict_in["signal_values_ext"]


        # run simulation
        net_osim(osim_file=self.osim_file, step_size=self.step_size, n_steps=self.n_steps, nn_file_name=self.nn_file_name, net_model_file=net_model_file, include=self.include,
                 controls_dict=self.controls_dict, save_folder=self.save_folder, nn_init=self.nn_init, integ_acc=self.integ_acc, perturb=self.perturb,  gravity=self.gravity, markers=None, 
                 coord_plot=self.coord_plot, musc_acti=self.musc_acti, rates_plot=None, visualize=self.visualize, 
                 show_plot=False, save_kin=True)
        
        #return sh_pos, elb_pos
        
        

if __name__ == '__main__':
    main()
