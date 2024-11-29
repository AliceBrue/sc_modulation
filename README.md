# sc_model_farms

sc_model_farms is a repository to build SC networks using FARMS, integrate them with OpenSim models, run simulations and optimise neural inputs.

## Dependencies

- numpy, matplotlib, flatten_dict, networkx, pandas, scipy
- OpenSim with python wrapping: https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python
- FARMS pylog, network and container libraries are already in the repom to install each: cd DIR, pip install -e . --user (https://gitlab.com/farmsim)

## Usage

### Folders:
- models/: OpenSim and SC models
- data/: recording data
- results/: simulations results 

### Main scripts:
- optim_brain_input.py: run optimisation of the brain inputs for various trajectories, with or without minimal SC
- optim_sc_weight_input.py: run optimisation of the SC weights for various trajectories
- run_net_sim_groups_input.py: build SC networks with various pathways and weights and run OpenSim simulations
- run_net_sim_pairs.py: build SC networks with Ia_MN, Ia_In pair and various weights and run OpenSim simulations
- plot_optim_target.py: plot target and optimisation kinematics together
- plot_input.py: plot optimised brain inputs
- metrics_osim_input_rmse.py: compute various metrics for simulation results
- plot_metrics.py: plot metrics for various scenarios
- sensitivity_analysis.py: run sensitivity analysis for all SC pathways on movement smoothness and deviation to perturbation

### Tool functions:
- brain_inputs.py: functions to define sinusoidal brain inputs
- connections.py : functions to build SC networks
- convert_data.py: functions to extract and align recording and simulation data
- EMG_activation.py: functions to compute EMG activation time windows
- gen_net.py: functions to generate SC networks using FARMS
- metrics.py: functions to compute metrics
- net_osim.py: functions to integrate and simulate SC networks with OpenSim models
- opensim_environment.py: environment to simulate OpenSim models
- optim_algo_input.py: functions to define brain inputs optimisation algorithms
- optim_algo.py: functions to define weights optimisation algorithms
- osim_model.py: functions to modify OpenSim model properties


