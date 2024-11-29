""" farms_container """
import os

from .container import Container


def get_include():
    """ Get include paths for pxd """
    return os.path.dirname(os.path.abspath(__file__))
