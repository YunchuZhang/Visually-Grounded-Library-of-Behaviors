import os
import re
import socket
import getpass
import yaml
import importlib
from addict import Dict
from copy import deepcopy
from xml.etree import ElementTree as et


def config_from_yaml_file(filename):
    config = Dict(load_yaml(filename))
    return config

def load_yaml(filename):
    with open(filename, 'r') as f:
        content = yaml.load(f, Loader=yaml.Loader)
    return content

def get_query_dir(query_dir):
    hostname = socket.gethostname()
    username = getpass.getuser()
    paths_yaml_fn = os.path.join(get_source_dir(), 'paths.yaml')
    with open(paths_yaml_fn, 'r') as f: 
        paths_config = yaml.load(f, Loader=yaml.Loader)

    for hostname_re in paths_config:
        if re.compile(hostname_re).match(hostname) is not None:
            for username_re in paths_config[hostname_re]:
                if re.compile(username_re).match(username) is not None:
                    return paths_config[hostname_re][username_re][query_dir]

    raise Exception('No matching hostname or username in config file')

def get_baseline_dir():
    return get_query_dir('baseline_dir')

def get_6dof_graspnet_dir():
    return get_query_dir('6dof_graspnet_dir')

def get_grnn_dir():
    return get_query_dir('grnn_dir')

def maybe_refine_mesh_path(mesh_path):
    if os.path.exists(mesh_path):
        return mesh_path
    else:
        source_mesh_dir = get_mesh_dir()
        remove_str = '../meshes/'
        new_path = os.path.join(source_mesh_dir,
                mesh_path[len(remove_str):])
        return new_path

def get_gym_dir():
	return get_query_dir('gym_dir')

def get_root_dir():
    return get_query_dir('root_dir')

def get_mesh_root_dir():
    return get_query_dir('mesh_root_dir')

def get_mesh_dir():
    return os.path.join(get_query_dir('mesh_root_dir'), 'meshes')

def get_object_xml_dir():
    return os.path.join(get_query_dir('mesh_root_dir'), 'xmls')


def get_source_dir():
    return os.getenv("GPV_SOURCE_DIR")

def makedir(path):
    if not os.path.exists(path):
        print(f"creating new path: {path}")
        os.makedirs(path)

def import_class_from_config(spec):
    assert "fn" in spec, f"missing 'fn' in {spec}"
    fn_class = import_function(spec['fn'])

    params = spec.get('params') or {}
    fn_config = fn_class.Config().update(params)
    return fn_class, fn_config

def import_function(fn_name):
    fn_path, fn_name = fn_name.split(":")
    module = importlib.import_module(fn_path)
    fn = getattr(module, fn_name)
    return fn

class Config:
    def __init__(self, **kwargs):
        for key, value in self.__class__.__dict__.items():
            if key.startswith('__'):
                continue
            elif key in kwargs:
                value = kwargs[key]
            else:
                value = deepcopy(value)
            self.__dict__[key] = value

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def update(self, params_dict):
        for key, value in self.__dict__.items():
            if key in params_dict:
                setattr(self, key, params_dict[key])
        return self


