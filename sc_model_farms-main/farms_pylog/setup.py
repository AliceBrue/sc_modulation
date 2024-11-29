import setuptools

setuptools.setup(
    name='farms_pylog',
    version='0.1',
    description='Python logger for farms framework',
    url='https://gitlab.com/FARMSIM/farms_pylog.git',
    author='biorob-farms',
    author_email='biorob-farms@groupes.epfl.ch',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['colorama'],
    zip_safe=False
)
