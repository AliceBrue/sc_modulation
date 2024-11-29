# sc_model_farms

sc_model_farms is a repository to build SC nertworks using FARMS, integrate them with OpenSim models and run simulations.

## Dependencies

- numpy, matplotlib, flatten_dict, networkx
- OpenSim with python wrapping: https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python
- FARMS pylog, network and container libraries are already in the repom to install each: cd DIR, pip install -e . --user (https://gitlab.com/farmsim)

## Usage

- run_net_sim.py: main script to build SC nertworks and run OpenSim simulations
- gen_net.py: functions to build SC models using FARMS
- net_osim.py: functions to integrate and simulate SC networks with OpenSim models
- opensim_environment.py: environement to integrate OpenSim models
- osim_model.py: functions to modify OpenSim models
- models/: OpenSim and SC models
- results/: simulations results 
