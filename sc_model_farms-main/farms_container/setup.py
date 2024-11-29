import numpy
import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name='farms_container',
    version='0.1',
    description='Data container class for farms experiments',
    url='https://gitlab.com/FARMSIM/farms_container.git',
    author='biorob-farms',
    author_email='biorob-farms@groupes.epfl.ch',
    license='Apache-2.0',
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=[
        'numpy',
        'Cython',
        'farms_pylog @ git+https://gitlab.com/FARMSIM/farms_pylog.git',
        'pandas',
    ],
    ext_modules=cythonize("farms_container/*.pyx"),
    zip_safe=False,
    package_data = {
        'farms_container': ['*.pxd'],
    },
    include_dirs=[numpy.get_include()]
)
