# farms_container

Repository contains wrapper classes for storing data as tables.

# Installation

## Requirements

- Code is only currently verified with Python3

- The installation requires Cython. To install Cython,

    `pip install cython`


(**NOTE** : *Depending on your system installation you may want to use pip3 instead of pip to use python3*)

## Install

- Navigate to the root of the directory
    `$ cd farms_container`

- Install system wide with pip
    `$ pip install .`

- Install for user with pip
    `$ pip install . --user`

- Install in developer mode so that you don't have to install every time you make
changes to the repository
    `$ pip install -e .`
    - (You may use `--user` option with the above command if you want install only for a user)

- To only compile the module
    `$ python setup.py build_ext -i`

# Usage

```python
from farms_container import Container
#: Create an instance of container
container = Container() #: Note that container is a singleton class!!
#: Add a new namespace to the container
x1 = container.add_namespace('x1')
#: Add two new table under the namespace
table1 = x1.add_table('table_x1')
table2 = x1.add_table('table_x2')
#: Add new parameter under namespace x1 in table 1
(param1, value) = table1.add_parameter('p1')
#: Add new parameter under namespace x1 in table 2
(param2, value) = table2.add_parameter('p1')
#: param1/param2 are objects of type farms_container::Parameter.
#: They give you direct access to the data
#: Before accessing you need initialize the data tables
container.initialize()
#: You can update the logs
container.update_log()
```
