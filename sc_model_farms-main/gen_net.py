""" Generate network from SC model """

import networkx as nx
import os
from matplotlib import pyplot as plt

from farms_container import Container
from farms_network.neural_system import NeuralSystem
import numpy as np
import farms_pylog as pylog
#import control

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=12.0)     # fontsize of the axes title
plt.rc('axes', labelsize=12.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12.0)    # fontsize of the tick labels


def gen_net_file(file_path, muscles=['DELT1', 'TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA'],
                 fcts=['elb_flex', 'elb_ext', 'elb_ext', 'elb_ext', 'elb_flex', 'elb_flex', 'elb_flex'],
                 Ia_antagonists=['TRIlong', 'DELT1,BIClong', 'BICshort,BRA', 'BICshort,BRA', 'TRIlong',
                                 'TRIlat,TRImed', 'TRIlat,TRImed'], Ia_delays=30, Ia_w=1, IaIn_w=1, IaIn_IaIn_w=1, 
                 Mn_Rn_w=1, Rn_Mn_w=1, Rn_IaIn_w=1, Rn_Rn_w=1, II_delays=30, II_w=1):
    """ Generate network description file """

    if os.path.exists(file_path):
        os.remove(file_path)
    file = open(file_path, "w+")
    file.write('Muscles type Ia_antagonist Ia_delay Ia_w IaIn_w IaIn_IaIn_w Mn_Rn_w Rn_Mn_w Rn_IaIn_w Rn_Rn_w II_delay'
               ' II_w\n')
    Ia_delays = Ia_delays*np.ones(len(muscles))
    Ia_w = Ia_w*np.ones(len(muscles))
    IaIn_w = IaIn_w*np.ones(len(muscles))
    IaIn_IaIn_w = IaIn_IaIn_w*np.ones(len(muscles))
    Mn_Rn_w = Mn_Rn_w*np.ones(len(muscles))
    Rn_Mn_w = Rn_Mn_w*np.ones(len(muscles))
    Rn_IaIn_w = Rn_IaIn_w*np.ones(len(muscles))
    Rn_Rn_w = Rn_Rn_w*np.ones(len(muscles))
    II_delays = II_delays*np.ones(len(muscles))
    II_w = II_w*np.ones(len(muscles))
    for i in range(len(muscles)):
        file.write(muscles[i] + ' ' + fcts[i] + ' ' + Ia_antagonists[i] + ' ' + str(Ia_delays[i]) + ' ' + str(Ia_w[i])
                   + ' ' + str(IaIn_w[i]) + ' ' + str(IaIn_IaIn_w[i]) + ' ' + str(Mn_Rn_w[i]) + ' ' + str(Rn_Mn_w[i]) +
                   ' ' + str(Rn_IaIn_w[i]) + ' ' + str(Rn_Rn_w[i]) + ' ' + str(II_delays[i]) + ' ' + str(II_w[i])
                   + '\n')
    file.close()


def gen_net(name, nn_file_name, net_model_file, step_size, model='leaky', tau=0.1, bias=-2.75, D=1, c_m=10,
            g_leak=2.8, include=[], save_folder='results_net/', legend=True, show_plot=False):
    """ Generate network """

    #: SC model
    sc_model = build_sc(net_model_file, step_size)

    #: Motorneurons
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

    return sc_model


class Muscle(object):
    """ Muscle properties """

    def __init__(self, name, step_size, type=None, Ia_antagonists=None, Ia_delay=0, Ia_w=1, IaIn_w=[1], IaIn_IaIn_w=1,
                 Mn_Rn_w=1, Rn_Mn_w=1, Rn_IaIn_w=1, Rn_Rn_w=1, II_delay=30, II_w=1, past_fiber_l=None):
        self.name = name
        self.type = type
        self.Ia_delay = Ia_delay
        self.Ia_antagonists = Ia_antagonists
        self.Ia_w = Ia_w
        if Ia_antagonists is not None and len(IaIn_w) == 1:
            self.IaIn_w = IaIn_w*np.ones(len(Ia_antagonists))
        else:
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
        #self.past_fiber_d = np.zeros(10)
        #self.transfer = control.TransferFunction([1, 11.4, 4.4], [1, 0.8], dt=step_size)


    def Prochazka_Ia_rates(self, model, muscle, a=4.3, b=2, c=10):
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


class SpinalCord(object):
    """ Spinal cord properties """

    def __init__(self, ext_muscles=None, flex_muscles=None):
        self.ext_muscles = ext_muscles
        self.flex_muscles = flex_muscles


def build_sc(sc_model_file, step_size):
    """ Build spinal cord model """

    file = open(sc_model_file, "r")
    lines = file.readlines()
    sc_model = SpinalCord()
    ext_muscles = {}
    flex_muscles = {}
    for i in range(1, len(lines)):
        muscle = lines[i].split(' ')
        if muscle[1].split("_")[0] not in ext_muscles.keys():
            ext_muscles[muscle[1].split("_")[0]] = {}
            flex_muscles[muscle[1].split("_")[0]] = {}
        if muscle[2].split(',')[0] in ['', 'None']:
            Ia_antagonists = None
        else:
            Ia_antagonists = muscle[2].split(',')
        if muscle[1].split("_")[-1] == 'ext':
            ext_muscles[muscle[1].split("_")[0]][muscle[0]] = Muscle(muscle[0], step_size, type=muscle[1],
                                                                     Ia_antagonists=Ia_antagonists,
                                                                     Ia_delay=float(muscle[3]), Ia_w=float(muscle[4]),
                                                                     IaIn_w=list(map(float, muscle[5].split(','))),
                                                                     IaIn_IaIn_w=float(muscle[6]),
                                                                     Mn_Rn_w=float(muscle[7]), Rn_Mn_w=float(muscle[8]),
                                                                     Rn_IaIn_w=float(muscle[9]),
                                                                     Rn_Rn_w=float(muscle[10]), II_delay=float(muscle[11]),
                                                                     II_w=float(muscle[12]))
        elif muscle[1].split("_")[-1] == 'flex':
            flex_muscles[muscle[1].split("_")[0]][muscle[0]] = Muscle(muscle[0], step_size, type=muscle[1],
                                                                      Ia_antagonists=Ia_antagonists,
                                                                      Ia_delay=float(muscle[3]), Ia_w=float(muscle[4]),
                                                                      IaIn_w=list(map(float, muscle[5].split(','))),
                                                                      IaIn_IaIn_w=float(muscle[6]),
                                                                      Mn_Rn_w=float(muscle[7]), Rn_Mn_w=float(muscle[8]),
                                                                      Rn_IaIn_w=float(muscle[9]), Rn_Rn_w=float(muscle[10]),
                                                                      II_delay=float(muscle[11]),
                                                                      II_w=float(muscle[12]))

    sc_model.ext_muscles = ext_muscles
    sc_model.flex_muscles = flex_muscles

    return sc_model


class Motorneurons(object):
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
                if 'IaIn' in include:
                    self.net.add_node(self.name + '_IaIn_' + muscle,
                                      model=model, tau=tau, bias=bias, D=D,
                                      x=float(_pos[j]) + anchor_x,
                                      y=y[0],
                                      color=color[1])
                if 'II' in include:
                    self.net.add_node(self.name + '_IIIn_' + muscle,
                                      model=model, tau=tau, bias=bias, D=D,
                                      x=float(_pos[j]) + anchor_x - 0.25,
                                      y=y[0],
                                      color=color[1])
                if 'Rn' in include:
                    y_Rn = 1.0 + anchor_y
                    if rev_pos:
                        y_Rn = -1.0 + anchor_y
                    self.net.add_node(self.name + '_Rn_' + muscle,
                                      model=model, tau=tau, bias=bias, D=D,
                                      x=float(_pos[j]) + anchor_x ,
                                      y=y_Rn,
                                      color=color[2])
        else:
            for j, muscle in enumerate(muscles.keys()):
                self.net.add_node(self.name + '_Mn_' + muscle,
                                  model=model, c_m=c_m, g_leak=g_leak,
                                  x=float(_pos[j]) + anchor_x,
                                  y=0.0 + anchor_y,
                                  color=color[0])
                if 'IaIn' in include:
                    self.net.add_node(self.name + '_IaIn_' + muscle,
                                      model=model, c_m=c_m, g_leak=g_leak,
                                      x=float(_pos[j]) + anchor_x,
                                      y=y[0],
                                      color=color[1])
                if 'II' in include:
                    self.net.add_node(self.name + '_IIIn_' + muscle,
                                      model=model, c_m=c_m, g_leak=g_leak,
                                      x=float(_pos[j]) + anchor_x - 0.25,
                                      y=y[0],
                                      color=color[1])
                if 'Rn' in include:
                    y_Rn = 1.0 + anchor_y
                    if rev_pos:
                        y_Rn = -1.0 + anchor_y
                    self.net.add_node(self.name + '_Rn_' + muscle,
                                      model=model, c_m=c_m, g_leak=g_leak,
                                      x=float(_pos[j]) + anchor_x,
                                      y=y_Rn,
                                      color=color[2])

    def add_connections(self, muscles, include):
        """ Connect the neurons."""

        if 'II' in include:
            for muscle in muscles.keys():
                self.net.add_edge(self.name + '_IIIn_' + muscle,
                                  self.name + '_Mn_' + muscle,
                                  weight=muscles[muscle].II_w)
        if 'Rn' in include:
            for muscle in muscles.keys():
                self.net.add_edge(self.name + '_Mn_' + muscle,
                                  self.name + '_Rn_' + muscle,
                                  weight=muscles[muscle].Mn_Rn_w)
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

    def __init__(self, name, muscles, include, anchor_x=0.0, anchor_y=0.0, color=['y', 'k'], rev_pos=False):
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
        _pos = np.arange(-_num_muscles, _num_muscles, 2.)

        y = [2.0 + anchor_y, 0.0 + anchor_y]
        yc = y
        f = 1
        if rev_pos:
            y = [-2.0 + anchor_y, 0.0 + anchor_y]
            yc = np.flip(yc)
            f = -1
        for j, muscle in enumerate(muscles.keys()):
            if "Ia" in include:
                # Ia to Mn
                self.afferents.add_node(self.name + '_' + muscle + '_Ia',
                                        model='sensory',
                                        x=float(_pos[j]) + anchor_x - 1.,
                                        y=y[0],
                                        color=color[0],
                                        init=0.0)
                # Ia to IaIn (same input as previous Ia but no additional PI)
                self.afferents.add_node(self.name + '_' + muscle + '_IaIn',
                                        model='sensory',
                                        x=float(_pos[j]) + anchor_x - 1.,
                                        y=y[0],
                                        color=color[0],
                                        init=0.0)

            if 'II' in include:
                self.afferents.add_node(self.name + '_' + muscle + '_II',
                                        model='sensory',
                                        x=float(_pos[j]) + anchor_x - 1. - 0.25,
                                        y=y[0],
                                        color=color[0],
                                        init=0.0)
            # controls
            self.afferents.add_node(self.name + '_' + muscle + '_C',
                                    model='sensory',
                                    x=float(_pos[j]) + anchor_x,
                                    y=yc[1],
                                    color=color[1],
                                    init=0.0)
            pi = any("PI" in string for string in include)
            if pi:
                self.afferents.add_node(self.name + '_' + muscle + '_CIa',
                                    model='sensory',
                                    x=float(_pos[j]) + anchor_x-0.25,
                                    y=yc[1],
                                    color=color[1],
                                    init=0.0)
            if 'control_IaIn' in include:
                self.afferents.add_node(self.name + '_' + muscle + '_CIaIn',
                                    model='sensory',
                                    x=float(_pos[j]) + anchor_x+0.25,
                                    y=yc[1],
                                    color=color[1],
                                    init=0.0)
            if 'control_Rn' in include:
                self.afferents.add_node(self.name + '_' + muscle + '_CRn',
                                    model='sensory',
                                    x=float(_pos[j]) + anchor_x+0.5,
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
            pi = any("PI" in string for string in include)
            if pi:
                self.net.add_edge(self.name + '_' + muscle + '_CIa',
                                  self.name + '_' + muscle + '_Ia',
                                  weight=-1.0)
            if 'control_IaIn' in include:
                self.net.add_edge(self.name + '_' + muscle + '_CIaIn',
                                  self.name + '_IaIn_' + muscle,
                                  weight=1.0)
            if 'control_Rn' in include:
                self.net.add_edge(self.name + '_' + muscle + '_CRn',
                                  self.name + '_Rn_' + muscle,
                                  weight=-1.0)

            # afferents
            if 'Ia' in include:
                # Ia to Mn
                self.net.add_edge(self.name + '_' + muscle + '_Ia',
                                  self.name + '_Mn_' + muscle,
                                  weight=muscles[muscle].Ia_w)
                # Ia to IaIn
                if 'IaIn' in include:
                    self.net.add_edge(self.name + '_' + muscle + '_IaIn',
                                      self.name + '_IaIn_' + muscle,
                                      weight=1)
            if 'II' in include:
                self.net.add_edge(self.name + '_' + muscle + '_II',
                                  self.name + '_IIIn_' + muscle,
                                  weight=1)

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

        if 'IaIn' in include:
            for muscle in muscles.keys():
                if muscles[muscle].Ia_antagonists is not None:
                    for j, antagonist in enumerate(muscles[muscle].Ia_antagonists):
                        self.net.add_edge(self.name+'_IaIn_' + muscle,
                                          self.name+'_Mn_' + antagonist,
                                          weight=-muscles[muscle].IaIn_w[j])
                        if 'IaIn_IaIn' in include:
                            self.net.add_edge(self.name + '_IaIn_' + muscle,
                                              self.name + '_IaIn_' + antagonist,
                                              weight=-muscles[muscle].IaIn_IaIn_w)
                        if 'Rn_Rn' in include:
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