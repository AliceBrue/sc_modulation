#!/usr/bin/env python3

"""Unit testing of Dae Generator"""

import test_singleton
from farms_container import Container
import numpy as np

container = Container.get_instance()
print(container)

x1 = container.add_namespace('x1')
x2 = container.add_namespace('x2')
x3 = container.add_namespace('x3')
x4 = container.add_namespace('x4')
print(container)
print(Container.get_instance())

x1.add_table('parameters_x1')
x2.add_table('parameters_x2')
x3.add_table('parameters_x3')
x4.add_table('parameters_x4')

x1.parameters_x1.add_parameter('p1')
x2.parameters_x2.add_parameter('p1')
x3.parameters_x3.add_parameter('p1')
x4.parameters_x4.add_parameter('p1')

container.initialize()

for j in range(3):
    container.x1.parameters_x1.values = np.random.random((1,))
    # dae.update_log()

print(container.x1.parameters_x1.values[0])
container.x1.parameters_x1[0].value = 10.00034
print(container.x1.parameters_x1.values[0])
