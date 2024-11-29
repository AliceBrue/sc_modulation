""" functions to build SC networks """

import numpy as np
from net_osim import *

aff_delay = 30.0 # afferent delay in milliseconds

def IaMn_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['Ia']
    connections = ['Ia_Mn']
    include = nodes + connections
    net_model['Ia_Mn_w'] =w*ones_column # variable weight, fixed is already defined in gen_net
    net_model['Ia_delay'] = aff_delay*ones_column
    return net_model,include
    
def IaMn_syn_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['Ia']
    connections = ['Ia_Mn', 'Ia_synMn']
    include = nodes + connections
    #net_model['Ia_Mn_w'] = w*ones_column # variable weight 
    net_model['Ia_Mn_w_syn']= w*ones_column # variable weight
    net_model['Ia_delay'] = aff_delay*ones_column
    return net_model,include

def IaIn_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['Ia', 'IaIn']
    connections = ['Ia_Mn', 'Ia_IaIn', 'IaIn_Mn']
    include = nodes + connections
    net_model['Ia_In_w'] = ones_column # fixed weight 
    net_model['IaIn_Mn_w'] = w*ones_column # variable weight 
    net_model['Ia_delay'] = aff_delay*ones_column
    return net_model,include

def Ib_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['Ib', 'IbIn']
    connections = ["Ib_Mn", 'Ib_IbIn','IbIn_Mn']
    include = nodes + connections
    net_model['Ib_w'] = ones_column # fixed weights
    net_model['IbIn_w'] = w*ones_column # variable weights
    net_model['Ib_delay'] =  aff_delay*ones_column
    return net_model,include

def II_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['II', 'IIIn']
    connections = ["II_Mn", 'II_IIIn','IIIn_Mn']
    include = nodes + connections
    net_model['II_w'] = w*ones_column # variable weights
    net_model['II_delay'] = aff_delay*ones_column
    return net_model,include

def Rn_Mn_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['Rn']
    connections = ['Mn_Rn','Rn_Mn']
    include = nodes + connections
    net_model['Mn_Rn_w'] = ones_column # fixed weights
    net_model['Rn_Mn_w'] = w*ones_column # variable weights
    net_model['Ib_delay'] = aff_delay*ones_column
    return net_model,include

def Rn_In_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['Ia','Rn','IaIn']
    connections = ['Mn_Rn','Rn_Mn','Rn_IaIn', 'IaIn_Mn', 'Ia_IaIn', 'Ia_Mn']
    include = nodes + connections
    net_model['Mn_Rn_w'] = ones_column # fixed weights
    net_model['IaIn_Mn_w'] = ones_column # fixed weights
    net_model['Ia_In_w'] = ones_column # fixed weight 
    net_model['Ia_Mn_w'] = ones_column # fixed weight 
    net_model['Rn_IaIn_w'] = w*ones_column # variable weights
    net_model['Ia_delay'] = aff_delay*ones_column
    return net_model,include

def Rn_Rn_connections(net_model,w):
    ones_column = np.ones(len(net_model.index))
    nodes = ['Rn', 'Ia']
    connections = ['Mn_Rn','Rn_Mn','Rn_Rn', 'Ia_Mn']
    include = nodes + connections
    net_model['Mn_Rn_w'] = ones_column # fixed weights
    net_model['Rn_Mn_w'] = ones_column # fixed weights
    net_model['Ia_Mn_w'] = ones_column # fixed weights
    net_model['Rn_Rn_w'] = w*ones_column # variable weights
    net_model['Ia_delay'] = aff_delay*ones_column
    return net_model,include

def savings(group,net_model,w,save_folder):
    current_save = save_folder + 'group_' + group.split("_")[0] + '/' + group + '/' + group + '_' +  str(int(w*10)) + '/'
    if not os.path.isdir(save_folder + 'group_' + group.split("_")[0] + '/'):
        os.mkdir(save_folder + 'group_' + group.split("_")[0] + '/')
    if not os.path.isdir(save_folder + 'group_' + group.split("_")[0] + '/' + group + '/'):
        os.mkdir(save_folder + 'group_' + group.split("_")[0] + '/' + group + '/')
    if not os.path.isdir(current_save):
        os.mkdir(current_save)
    net_model.to_excel(current_save + 'net_model_' + group + '_' + str(int(w*10)) +  '.xlsx')
    return current_save

def select_connections(group, net_model, w, save_group=False, save_folder=None):
    if group == 'Ia_Mn':
        net_model,include = IaMn_connections(net_model,w)
        rates_plot = ['all', ['Mn', 'Ia']]
    elif group == 'Ia_Mn_syn':
        net_model,include = IaMn_syn_connections(net_model,w)
        rates_plot = ['all', ['Mn', 'Ia']]
    elif group == 'Ia_In':
        net_model,include = IaIn_connections(net_model,w)
        rates_plot = ['all', ['Mn','Ia', 'IaIn']]
    elif group == 'Ib':
        net_model,include = Ib_connections(net_model,w)
        rates_plot = ['all', ['Mn','Ib']]
    elif group == 'II':
        net_model,include = II_connections(net_model,w)
        rates_plot = ['all', ['Mn','II']]
    elif group == 'Rn_Mn':
        net_model,include = Rn_Mn_connections(net_model,w)
        rates_plot = ['all', ['Mn','Rn']]
    elif group == 'Rn_In':
        net_model,include = Rn_In_connections(net_model,w)
        rates_plot = ['all', ['Mn','Rn','Ia', 'IaIn']]
    elif group == 'Rn_Rn':
        net_model,include = Rn_Rn_connections(net_model,w)
        rates_plot = ['all', ['Mn','Rn']]
    else:
        sys.exit("Error, check the group list for missing or wrong arguments")
    if save_group:
        current_save = savings(group,net_model,w,save_folder)
    else :
        current_save = None
    return net_model,include, current_save,rates_plot

