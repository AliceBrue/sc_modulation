from farms_container.container import Container
import numpy as np
container = Container(max_iterations=100)
container.add_namespace('test')
t1 = container.test.add_table('table1')
(p1, _) = container.test.table1.add_parameter('p1', 0.1)
container.initialize()
print(p1.value)
for j in range(1, 100):
    t1.values = np.array([np.double(j),])
    container.update_log()
    print(np.asarray(t1.values))
