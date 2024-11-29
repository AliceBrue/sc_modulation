from farms_dae_generator.dae_generator import DaeGenerator
import numpy as np
import sympy as sm
import farms_pylog as pylog
from sympy.utilities.codegen import codegen
pylog.set_level("debug")


def main():
    """Main function"""
    #: DaeGenerator
    dae = DaeGenerator()

    #: States
    theta = dae.add_x('theta', 0.0, param_type='sym')
    theta_dot = dae.add_x('d_theta', 0.0, param_type='sym')

    #: Constants
    mass = dae.add_c('m', 1., param_type='val')
    gravity = dae.add_c('g', 9.81, param_type='val')
    length = dae.add_c('l', 1., param_type='val')

    #: ode
    theta_double_dot = dae.add_ode('theta_double_dot',
                                   -mass.param*gravity.param*length.param*sm.sin(
                                       theta.param))
    theta_dot = dae.add_ode('theta_dot', theta.param)

    rhs_of_odes = sm.Matrix([theta_double_dot.param, theta_dot.param])

    pylog.debug(rhs_of_odes.jacobian(dae.x.get_all_sym()))

    [(cf, cs), (hf, hs)] = codegen(('c_odes', rhs_of_odes), language='c')

    pylog.info(cs)


if __name__ == '__main__':
    main()
