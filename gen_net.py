""" functions to generate SC networks using FARMS """

import networkx as nx
import os
from matplotlib import pyplot as plt
import pandas as pd
from farms_container import Container
from farms_network.neural_system import NeuralSystem
import numpy as np
import farms_pylog as pylog


def gen_net(name, nn_file_name, net_model_file, step_size, model='leaky', tau=0.1, bias=-0.5, D=8, c_m=10,
            g_leak=2.8, include=[], save_folder='results_net/', legend=True, show_plot=False):  # save_folder should not be by default
    """ Generate network """

    #: SC model
    #sc_model = build_sc(net_model_file, step_size)
    sc_model = build_sc_xlsx(net_model_file, step_size)
    #: Motorneurons - all neurons depending on what is in include
    ext_net_Mn = {}
    flex_net_Mn = {}
    for j in range(len(sc_model.ext_muscles.keys())):
        joint = list(sc_model.ext_muscles.keys())[j]
        ext_net_Mn[joint] = Motorneurons(name, sc_model.ext_muscles[joint], model, tau, bias, D, c_m, g_leak,
                                         include, anchor_x=0.+j*20, anchor_y=2.)
        flex_net_Mn[joint] = Motorneurons(name, sc_model.flex_muscles[joint], model, tau, bias, D, c_m, g_leak,
                                          include, anchor_x=0.+j*20, anchor_y=8., rev_pos=True)
    #: Sensory Afferents
    ext_net_afferents = {}
    flex_net_afferents = {}
    for j in range(len(sc_model.ext_muscles.keys())):
        joint = list(sc_model.ext_muscles.keys())[j]
        ext_net_afferents[joint] = Afferents(name, sc_model.ext_muscles[joint], include, 
                                             anchor_x=0.+j*20, anchor_y=0)
        flex_net_afferents[joint] = Afferents(name, sc_model.flex_muscles[joint], include, 
                                              anchor_x=0.+j*20, anchor_y=8., rev_pos=True)

    #: Net
    ext_net = {}
    flex_net = {}
    net = {}
    for j in sc_model.ext_muscles.keys():
        ext_net[j] = ConnectAfferents2Mn(ext_net_Mn[j].net, ext_net_afferents[j].afferents, sc_model.ext_muscles[j],
                                         include=include)
        flex_net[j] = ConnectAfferents2Mn(flex_net_Mn[j].net, flex_net_afferents[j].afferents, sc_model.flex_muscles[j],
                                          include=include)
        sc_model.ext_muscles[j].update(sc_model.flex_muscles[j])
        net[j] = ConnectAntagonists(ext_net[j].net, flex_net[j].net, sc_model.ext_muscles[j], include)


    if len(sc_model.ext_muscles.keys()) == 1:
        full_net = nx.compose_all([net[list(sc_model.ext_muscles.keys())[0]].net])
    else:
        full_net = net[list(sc_model.ext_muscles.keys())[0]]
        for j in range(1, len(sc_model.ext_muscles.keys())):
            full_net = ConnectNets(full_net.net, net[list(sc_model.ext_muscles.keys())[j]].net)
        full_net = nx.compose_all([full_net.net])

    #: Location to save the networks
    net_dir = os.path.join(
        os.path.dirname(__file__),
        save_folder + nn_file_name + '.graphml')
    try:
        nx.write_graphml(full_net, net_dir)
    except IOError:
        if not os.path.isdir(os.path.split(net_dir)[0]):
            pylog.info('Creating directory : {}'.format(net_dir))
            os.mkdir(os.path.split(net_dir)[0])
            nx.write_graphml(full_net, net_dir)
        else:
            pylog.error('Error in creating directory!')
            raise IOError()

    if len(sc_model.ext_muscles.keys()) > 1:
        for j in sc_model.ext_muscles.keys():
            net[j] = nx.compose_all([net[j].net])
            net_dir = os.path.join(
                os.path.dirname(__file__),
                save_folder + j + '_' + nn_file_name + '.graphml')
            try:
                nx.write_graphml(net[j], net_dir)
            except IOError:
                if not os.path.isdir(os.path.split(net_dir)[0]):
                    pylog.info('Creating directory : {}'.format(net_dir))
                    os.mkdir(os.path.split(net_dir)[0])
                    nx.write_graphml(net[j], net_dir)
                else:
                    pylog.error('Error in creating directory!')
                    raise IOError()

    #: Visualize network using Matplotlib
    # CONTAINER
    container = Container()

    # #: Initialize network
    net_ = NeuralSystem(
        os.path.join(
            os.path.dirname(__file__),
            save_folder + nn_file_name + '.graphml'),
        container)
    if len(sc_model.ext_muscles.keys()) > 1:
        legend = False
    net_.visualize_network(node_size=50,
                           node_labels=False,
                           edge_labels=False,
                           edge_alpha=True,
                           legend=legend,
                           include=include,
                            plt_out=plt)
    print(save_folder + nn_file_name)
    plt.savefig(save_folder + nn_file_name)

    if len(sc_model.ext_muscles.keys()) > 1:
        for j in sc_model.ext_muscles.keys():
            # CONTAINER
            container = Container()

            # #: Initialize network
            net_ = NeuralSystem(
                os.path.join(
                    os.path.dirname(__file__),
                    save_folder + j + '_' + nn_file_name +'.graphml'),
                container)
            net_.visualize_network(node_size=50,
                                   node_labels=False,
                                   edge_labels=False,
                                   edge_alpha=True,
                                   legend=True,
                                   include=include,
                                   plt_out=plt)
            plt.savefig(save_folder + j + '_' + nn_file_name)

    if show_plot:
       plt.show()
       #plt.close()
      
    

    return sc_model


class Muscle(object):
    """ Muscle properties """

    def __init__(self, name, step_size, type=None, Ia_antagonists=None,Ia_synergistics=None, Ia_delay=0, Ia_Mn_w=0, Ia_In_w=0, IaIn_w=[0], IaIn_IaIn_w=1,
                 Mn_Rn_w=1, Rn_Mn_w=1, Rn_IaIn_w=1, Rn_Rn_w=1, II_delay=30, II_w=1, Ib_delay=0, Ib_w=1, IbIn_w=1, Ia_Mn_syn_w =1, past_fiber_l=None):
        self.name = name
        self.type = type
        self.Ia_delay = Ia_delay
        self.Ia_antagonists = Ia_antagonists
        """if Ia_antagonists is not None and len(IaIn_w) == 1:
            self.IaIn_w = IaIn_w*np.ones(len(Ia_antagonists))
        else:
            self.IaIn_w = IaIn_w"""
        # Checker pourquoi pb antagonist
        self.Ia_synergistics = Ia_synergistics  
        self.Ia_Mn_w = Ia_Mn_w
        self.Ia_In_w = Ia_In_w
        self.IaIn_w = IaIn_w
        self.IaIn_IaIn_w = IaIn_IaIn_w
        self.Mn_Rn_w = Mn_Rn_w
        self.Rn_Mn_w = Rn_Mn_w
        self.Rn_IaIn_w = Rn_IaIn_w
        self.Rn_Rn_w = Rn_Rn_w
        self.past_fiber_l = past_fiber_l
        self.past_Ia_rates = np.zeros(max(1, int(self.Ia_delay / (step_size * 1000))))
        self.II_delay = II_delay
        self.II_w = II_w
        self.past_II_rates = np.zeros(max(1, int(self.II_delay / (step_size * 1000))))
        self.Ib_delay = Ib_delay
        self.Ib_w = Ib_w
        self.IbIn_w = IbIn_w
        self.Ia_Mn_syn_w = Ia_Mn_syn_w
        self.past_Ib_rates = np.zeros(max(1, int(self.Ib_delay / (step_size * 1000))))
        #self.past_fiber_d = np.zeros(10)
        #self.transfer = control.TransferFunction([1, 11.4, 4.4], [1, 0.8], dt=step_size)


    def Prochazka_Ia_rates(self, model, muscle, a=4.3, b=2, c=10):  # muscle = OpenSim objet
        """ Compute Prochazka Ia rates """

        opt_l = muscle.getOptimalFiberLength() * 1000
        max_v = muscle.getMaxContractionVelocity() * opt_l
        fiber_l = muscle.getFiberLength(model.state) * 1000
        if self.past_fiber_l is None:
            self.past_fiber_l = opt_l
            fiber_v = 0.001
        else:
            fiber_v = (fiber_l - self.past_fiber_l) / model.step_size
        self.past_fiber_l = fiber_l
        rate = a * np.sign(fiber_v) * np.exp(0.6 * np.log(max(min(abs(fiber_v), max_v), 0.01))) + \
               b * (min(fiber_l, 1.5 * opt_l) - opt_l) + c
        norm_rate = max(rate / (a * np.exp(0.6 * np.log(max_v)) + b * 0.5 * opt_l + c), 0)
        
        #: Update past rates
        self.past_Ia_rates[:-1] = self.past_Ia_rates[1:]
        self.past_Ia_rates[-1] = norm_rate


    def Prochazka_II_rates(self, model, muscle, a=13.5, b=10):
        """ Compute Prochazka II rates """

        opt_l = muscle.getOptimalFiberLength() * 1000
        fiber_l = muscle.getFiberLength(model.state) * 1000
        delta_l = min(abs(fiber_l-opt_l), opt_l)
        rate = a*delta_l + b
        norm_rate = rate / (a*opt_l + b)

        """self.past_fiber_d[:-1] = self.past_fiber_d[1:]
        self.past_fiber_d[-1] = delta_l
        _, rate = control.forced_response(self.transfer, np.arange(0, 0.1, 0.01), self.past_fiber_d,
                                                self.past_II_rates[0])
        norm_rate = rate[-1]"""

        #: Update past rates
        self.past_II_rates[:-1] = self.past_II_rates[1:]
        self.past_II_rates[-1] = norm_rate
        
        
    def Prochazka_Ib_rates(self, model, muscle, pf = 1):  # muscle = OpenSim objet
        """ Compute Crago Ib rates """

        tension = muscle.getFiberForce(model.state)
        tension_max = muscle.getMaxIsometricForce()
        norm_rate = pf*tension/tension_max
        
        #: Update past rates
        self.past_Ib_rates[:-1] = self.past_Ib_rates[1:]
        self.past_Ib_rates[-1] = norm_rate


class SpinalCord(object):
    """ Spinal cord properties """

    def __init__(self, ext_muscles=None, flex_muscles=None):
        self.ext_muscles = ext_muscles
        self.flex_muscles = flex_muscles

def build_sc_xlsx(sc_model_file, step_size):
    """Build spinal cord model"""
    
    muscle = pd.read_excel(sc_model_file)

    sc_model = SpinalCord()
    ext_muscles = {}
    flex_muscles = {}
    
    Ia_Mn_syn_weights = np.zeros(muscle.shape[0])
    for i in range(0, muscle.shape[0]):
        print(muscle.at[i,'type'].split("_"))
        if muscle.at[i,'type'].split("_")[0] not in ext_muscles.keys():
            ext_muscles[muscle.at[i,'type'].split("_")[0]] = {}
            flex_muscles[muscle.at[i,'type'].split("_")[0]] = {}
        if muscle.at[i,'Ia_antagonist'].split(',')[0] in ['', 'None']:
                Ia_antagonists = None
        else:
            Ia_antagonists = muscle.at[i,'Ia_antagonist'].split(',')
        if 'Ia_synergistic' in muscle.head():
            if muscle.at[i,'Ia_synergistic'].split(',')[0] in ['', 'None']:
                    Ia_synergistics = None
            else:
                Ia_synergistics = muscle.at[i,'Ia_synergistic'].split(',')
                Ia_Mn_syn_weights[i] = muscle.at[i, 'Ia_Mn_w_syn']
        else:
            Ia_synergistics = None
            
        if muscle.at[i,'type'].split("_")[-1] == 'ext':
            ext_muscles[muscle.at[i,'type'].split("_")[0]][muscle.at[i,'Muscles']] = Muscle(muscle.at[i,'Muscles'], step_size, type=muscle.at[i,'type'],
                                                                                            Ia_antagonists=Ia_antagonists,
                                                                                            Ia_synergistics=Ia_synergistics,
                                                                                            Ia_delay=float(muscle.at[i,'Ia_delay']), Ia_In_w=float(muscle.at[i,'Ia_In_w']),
                                                                                            Ia_Mn_w=float(muscle.at[i,'Ia_Mn_w']), IaIn_w=float(muscle.at[i,'IaIn_Mn_w']),
                                                                                            IaIn_IaIn_w=float(muscle.at[i,'IaIn_IaIn_w']),
                                                                                            Mn_Rn_w=float(muscle.at[i,'Mn_Rn_w']), Rn_Mn_w=float(muscle.at[i,'Rn_Mn_w']),
                                                                                            Rn_IaIn_w=float(muscle.at[i,'Rn_IaIn_w']),
                                                                                            Rn_Rn_w=float(muscle.at[i, 'Rn_Rn_w']), II_delay=float(muscle.at[i,'II_delay']),
                                                                                            II_w=float(muscle.at[i,'II_w']),
                                                                                            Ib_delay=float(muscle.at[i, 'Ib_delay']), Ib_w=float(muscle.at[i,'Ib_w']),
                                                                                            IbIn_w=float(muscle.at[i,'IbIn_w']),
                                                                                            Ia_Mn_syn_w= Ia_Mn_syn_weights[i] #float(muscle.at[i, 'Ia_Mn_syn_w'])
                                                                                            )
        elif muscle.at[i,'type'].split("_")[-1] == 'flex':
            flex_muscles[muscle.at[i,'type'].split("_")[0]][muscle.at[i,'Muscles']] = Muscle(muscle.at[i,'Muscles'], step_size, type=muscle.at[i,'type'],
                                                                                            Ia_antagonists=Ia_antagonists,
                                                                                            Ia_synergistics=Ia_synergistics,
                                                                                            Ia_delay=float(muscle.at[i,'Ia_delay']), Ia_In_w=float(muscle.at[i,'Ia_In_w']),
                                                                                            Ia_Mn_w=float(muscle.at[i,'Ia_Mn_w']), IaIn_w=float(muscle.at[i,'IaIn_Mn_w']),
                                                                                            IaIn_IaIn_w=float(muscle.at[i,'IaIn_IaIn_w']),
                                                                                            Mn_Rn_w=float(muscle.at[i,'Mn_Rn_w']), Rn_Mn_w=float(muscle.at[i,'Rn_Mn_w']),
                                                                                            Rn_IaIn_w=float(muscle.at[i,'Rn_IaIn_w']),
                                                                                            Rn_Rn_w=float(muscle.at[i, 'Rn_Rn_w']), II_delay=float(muscle.at[i,'II_delay']),
                                                                                            II_w=float(muscle.at[i,'II_w']),
                                                                                            Ib_delay=float(muscle.at[i, 'Ib_delay']), Ib_w=float(muscle.at[i,'Ib_w']),
                                                                                            IbIn_w=float(muscle.at[i,'IbIn_w']),
                                                                                            Ia_Mn_syn_w=Ia_Mn_syn_weights[i] #float(muscle.at[i, 'Ia_Mn_syn_w'])
                                                                                            )

    sc_model.ext_muscles = ext_muscles
    sc_model.flex_muscles = flex_muscles
    return sc_model


class Motorneurons(object):  # all neurons depending on what is in include
    """Motorneurons layers. Also contains interneurons."""

    def __init__(self, name, muscles, model, tau=0.1, bias=-2.75, D=1, c_m=10, g_leak=2.8,
                 include=[], anchor_x=0.0, anchor_y=0.0, color=['r', 'b', 'm'], rev_pos=False):
        super(Motorneurons, self).__init__()
        self.name = name
        self.net = nx.DiGraph()

        #: Methods
        self.add_neurons(muscles, model, tau, bias, D, c_m, g_leak, include, anchor_x, anchor_y, color, rev_pos)
        self.add_connections(muscles, include)

        return

    def add_neurons(self, muscles, model, tau, bias, D, c_m, g_leak, include, anchor_x, anchor_y, color, rev_pos):
        """ Add neurons. """

        pylog.debug("Adding motorneurons")
        _num_muscles = len(muscles)
        _pos = np.arange(-_num_muscles, _num_muscles, 2.)
        #position figure
        y = [2.0 + anchor_y, 0.0 + anchor_y]
        f = 1
        if rev_pos:
            y = [-2.0 + anchor_y, 0.0 + anchor_y]
            f = -1

        if model == 'leaky':
            for j, muscle in enumerate(muscles.keys()):
                self.net.add_node(self.name + '_Mn_' + muscle,
                                  model=model, tau=tau, bias=bias, D=D,
                                  x=float(_pos[j]) + anchor_x,
                                  y=0.0 + anchor_y,
                                  color=color[0])
                if 'IaIn_Mn' in include:
                    self.net.add_node(self.name + '_IaIn_' + muscle,
                                      model=model, tau=tau, bias=bias, D=D,
                                      x=float(_pos[j]) + anchor_x,
                                      y=y[0],
                                      color=color[1])
                if 'II_Mn' in include:
                    self.net.add_node(self.name + '_IIIn_' + muscle,
                                      model=model, tau=tau, bias=bias, D=D,
                                      x=float(_pos[j]) + anchor_x - 0.25,
                                      y=y[0],
                                      color=color[1])
                if 'IbIn' in include:
                    self.net.add_node(self.name + '_IbIn_' + muscle,
                                        model=model, tau=tau, bias=bias, D=D,
                                        x=float(_pos[j]) + anchor_x + 0.25,
                                        y=y[0],
                                        color=color[1])
                if 'Rn_Mn' in include:
                    y_Rn = 1.0 + anchor_y
                    if rev_pos:
                        y_Rn = -1.0 + anchor_y
                    self.net.add_node(self.name + '_Rn_' + muscle,
                                      model=model, tau=tau, bias=bias, D=D,
                                      x=float(_pos[j]) + anchor_x + 0.25,
                                      y=y_Rn,
                                      color=color[2])
        else:
            for j, muscle in enumerate(muscles.keys()):
                self.net.add_node(self.name + '_Mn_' + muscle,
                                  model=model, c_m=c_m, g_leak=g_leak,
                                  x=float(_pos[j]) + anchor_x,
                                  y=0.0 + anchor_y,
                                  color=color[0])
                if 'IaIn_Mn' in include:
                    self.net.add_node(self.name + '_IaIn_' + muscle,
                                      model=model, c_m=c_m, g_leak=g_leak,
                                      x=float(_pos[j]) + anchor_x,
                                      y=y[0],
                                      color=color[1])
                if 'II_Mn' in include:
                    self.net.add_node(self.name + '_IIIn_' + muscle,
                                      model=model, c_m=c_m, g_leak=g_leak,
                                      x=float(_pos[j]) + anchor_x - 0.25,
                                      y=y[0],
                                      color=color[1])
                if 'IbIn' in include:
                    self.net.add_node(self.name + '_IbIn_' + muscle,
                                      model=model, c_m=c_m, g_leak=g_leak,
                                        x=float(_pos[j]) + anchor_x + 0.25,
                                        y=y[0],
                                        color=color[1])
                if 'Rn_Mn' in include:
                    y_Rn = 1.0 + anchor_y
                    if rev_pos:
                        y_Rn = -1.0 + anchor_y
                    self.net.add_node(self.name + '_Rn_' + muscle,
                                      model=model, c_m=c_m, g_leak=g_leak,
                                      x=float(_pos[j]) + anchor_x + 0.25,
                                      y=y_Rn,
                                      color=color[2])

    def add_connections(self, muscles, include):
        """ Connect the neurons."""

        # IIIn to Mn
        if 'IIIn_Mn' in include:
            for muscle in muscles.keys():
                self.net.add_edge(self.name + '_IIIn_' + muscle,
                                  self.name + '_Mn_' + muscle,
                                  weight=muscles[muscle].II_w)
                
        # IbIn to Mn        
        if 'IbIn_Mn' in include:
            for muscle in muscles.keys():
                self.net.add_edge(self.name + '_IbIn_' + muscle,
                                  self.name + '_Mn_' + muscle,
                                  weight=-muscles[muscle].IbIn_w)
                
        if 'Rn_Mn' in include:
            for muscle in muscles.keys():
                if 'Mn_Rn' in include:
                    self.net.add_edge(self.name + '_Mn_' + muscle,
                                    self.name + '_Rn_' + muscle,
                                    weight=muscles[muscle].Mn_Rn_w)
                if 'Rn_Mn' in include:
                    self.net.add_edge(self.name + '_Rn_' + muscle,
                                    self.name + '_Mn_' + muscle,
                                    weight=-muscles[muscle].Rn_Mn_w)
                if 'Rn_IaIn' in include:
                    self.net.add_edge(self.name + '_Rn_' + muscle,
                                      self.name + '_IaIn_' + muscle,
                                      weight=-muscles[muscle].Rn_IaIn_w)
        return self.net


class Afferents(object):
    """Generate Afferents Network"""

    def __init__(self, name, muscles, include, anchor_x=0.0, anchor_y=0.0, color=['y', 'k', 'g'], rev_pos=False):
        """ Initialization. """
        super(Afferents, self).__init__()
        self.afferents = nx.DiGraph()
        self.name = name
        self.afferents.name = name

        #: Methods
        self.add_neurons(muscles, include, anchor_x, anchor_y, color, rev_pos)
        self.add_connections(muscles, include, anchor_x, anchor_y, color)
        return

    def add_neurons(self, muscles, include, anchor_x, anchor_y, color, rev_pos):
        """ Add neurons. """

        pylog.debug("Adding sensory afferents")
        _num_muscles = len(muscles)
        _pos = np.arange(-_num_muscles, _num_muscles, 2.)  # x vector for each "muscle"

        # position graph
        y = [2.0 + anchor_y, 0.0 + anchor_y]
        yc = y
        f = 1
        if rev_pos:
            y = [-2.0 + anchor_y, 0.0 + anchor_y]
            yc = np.flip(yc)
            f = -1
        for j, muscle in enumerate(muscles.keys()):
            if "Ia_Mn" in include:
                self.afferents.add_node(self.name + '_' + muscle + '_Ia',
                                        model='sensory',
                                        x=float(_pos[j]) + anchor_x - 1.,
                                        y=y[0],
                                        color=color[0],
                                        init=0.0)

            if 'II_Mn' in include:
                self.afferents.add_node(self.name + '_' + muscle + '_II',
                                        model='sensory',
                                        x=float(_pos[j]) + anchor_x - 1. - 0.25,
                                        y=y[0],
                                        color=color[0],
                                        init=0.0)
                
            if "Ib_Mn" in include:
                self.afferents.add_node(self.name + '_' + muscle + '_Ib',
                                        model='sensory',
                                        x=float(_pos[j]) + anchor_x - 1. + 0.25,
                                        y=y[0],
                                        color=color[2],
                                        init=0.0)
            # controls
            self.afferents.add_node(self.name + '_' + muscle + '_C',
                                    model='sensory',
                                    x=float(_pos[j]) + anchor_x,
                                    y=yc[1],
                                    color=color[1],
                                    init=0.0)
            # later later
            if 'control_IaIn' in include:
                self.afferents.add_node(self.name + '_' + muscle + '_CIaIn',
                                    model='sensory',
                                    x=float(_pos[j]) + anchor_x + 0.25,
                                    y=yc[1],
                                    color=color[1],
                                    init=0.0)
            if 'control_Rn' in include:
                self.afferents.add_node(self.name + '_' + muscle + '_CRn',
                                    model='sensory',
                                    x=float(_pos[j]) + anchor_x + 0.5,
                                    y=yc[1],
                                    color=color[1],
                                    init=0.0)

    def add_connections(self, muscles, include, anchor_x, anchor_y, color):

        return


class ConnectAfferents2Mn(object):
    """Connect Afferents to Mn"""

    def __init__(self, mn, afferents, muscles, include=[]):
        """ Initialization. """

        super(ConnectAfferents2Mn, self).__init__()
        self.net = nx.compose_all([mn, afferents])
        self.name = self.net.name
        #: Methods
        self.connect_circuits(muscles, include)
        return

    def connect_circuits(self, muscles, include):
        """ Connect Ia to MN and IaIn, and C to Mn"""

        for muscle in muscles.keys():
            # controls
            self.net.add_edge(self.name + '_' + muscle + '_C',
                                self.name + '_Mn_' + muscle,
                                weight=1.0)
            
            # later later
            if 'control_IaIn' in include:
                self.net.add_edge(self.name + '_' + muscle + '_CIaIn',
                                  self.name + '_IaIn_' + muscle,
                                  weight=1.0)
            
            if 'control_Rn' in include:
                self.net.add_edge(self.name + '_' + muscle + '_CRn',
                                  self.name + '_Rn_' + muscle,
                                  weight=-1.0)

            # afferents

            # Ia to Mn
            if 'Ia_Mn' in include:
                if 'Ia_synMn' in include:
                     self.net.add_edge(self.name + '_' + muscle + '_Ia',
                                    self.name + '_Mn_' + muscle,
                                    weight=muscles[muscle].Ia_Mn_w/(len(muscles[muscle].Ia_synergistics)))
                else:
                    self.net.add_edge(self.name + '_' + muscle + '_Ia',
                                    self.name + '_Mn_' + muscle,
                                    weight=muscles[muscle].Ia_Mn_w)
                
            # Ia to synergistic Mn    
            if 'Ia_synMn' in include:
                #if muscles[muscle].Ia_synergistics is not None:
                for j, synergistic in enumerate(muscles[muscle].Ia_synergistics):
                    self.net.add_edge(self.name + '_' + muscle + '_Ia',
                                        self.name+'_Mn_' + synergistic,
                                        weight=muscles[muscle].Ia_Mn_syn_w/(len(muscles[muscle].Ia_synergistics)))
                    
            # Ia to IaIn    
            if 'IaIn_Mn' in include:
                self.net.add_edge(self.name + '_' + muscle + '_Ia',
                                self.name + '_IaIn_' + muscle,
                                weight=muscles[muscle].Ia_In_w)
                
            # II to II In 
            if  'II_IIIn' in include:
                self.net.add_edge(self.name + '_' + muscle + '_II',
                                  self.name + '_IIIn_' + muscle,
                                  weight=1)

            # Ib to IbIn    
            if 'Ib_IbIn' in include:
                self.net.add_edge(self.name + '_' + muscle + '_Ib',
                                  self.name + '_IbIn_' + muscle,
                                  weight=muscles[muscle].Ib_w)

        return self.net


class ConnectAntagonists(object):
    """Connect Antagonist Muscles"""

    def __init__(self, ext_Mn, flex_Mn, muscles, include=[]):
        """ Initialization. """
        super(ConnectAntagonists, self).__init__()
        self.net = nx.compose_all([ext_Mn, flex_Mn])
        self.name = self.net.name
        #: Methods
        self.connect_circuits(muscles, include)
        return

    def connect_circuits(self, muscles, include):
        """ Connect IaIn to antagonists Mn"""
       
        for muscle in muscles.keys():
            if muscles[muscle].Ia_antagonists is not None:             
                for j, antagonist in enumerate(muscles[muscle].Ia_antagonists):
                    # Ia In to Mn
                    if 'IaIn_Mn' in include:
                        self.net.add_edge(self.name+'_IaIn_' + muscle,
                                        self.name+'_Mn_' + antagonist,
                                        #weight=-muscles[muscle].IaIn_w[j])
                                        weight=-muscles[muscle].IaIn_w)
                        if 'IaIn_IaIn' in include:
                            # Ia In to antagonist Ia In
                            self.net.add_edge(self.name + '_IaIn_' + muscle,
                                            self.name + '_IaIn_' + antagonist,
                                            weight=-muscles[muscle].IaIn_IaIn_w)
                    if 'Rn_Rn' in include:
                        # Rn to antagonist Rn
                        self.net.add_edge(self.name + '_Rn_' + muscle,
                                        self.name + '_Rn_' + antagonist,
                                        weight=-muscles[muscle].Rn_Rn_w)

        return self.net


class ConnectNets(object):
    """Connect Antagonist Muscles"""

    def __init__(self, net_Mn, net_Mn_2):
        """ Initialization. """
        super(ConnectNets, self).__init__()
        self.net = nx.compose_all([net_Mn, net_Mn_2])
        self.name = self.net.name
        self.connect_circuits()
        return

    def connect_circuits(self):
        """ """
        return self.net