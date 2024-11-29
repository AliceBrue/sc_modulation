""" Generate network from SC model"""

import os
from gen_net import *

def main():

    #: Model of interest
    net_model_file = 'models/SC_model_3D.xlsx'
    step_size = 0.01
    save_folder = 'results/test_net/'

    #: Network from sc model
    nn_file_name = 'net'
    neuron_model = 'leaky'  # 'lif_danner' more bio model but more param to definem check FARMS implementation
    tau = 0.001
    bias = -0.5
    D = 8
    #: SC pathways in: ['Ia_Mn', 'IaIn_Mn', 'IaIn_IaIn', 'II_IIIn', 'Ib_IbIn', 'Rn_Mn', 'Rn_IaIn', 'Rn_Rn']
    include = ['Ia_Mn', 'IaIn_Mn', 'IaIn_IaIn'] 
    
    #: Results folder
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
        
    gen_net('arm', nn_file_name, net_model_file, step_size, model=neuron_model, tau=tau, bias=bias, D=D, include=include,
            save_folder=save_folder, legend=True, show_plot=True)



if __name__ == '__main__':
        main()