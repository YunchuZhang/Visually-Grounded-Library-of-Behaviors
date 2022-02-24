import os
import re
import socket
import getpass
import yaml

def get_source_dir():
    return os.getenv("PYDISCO_ROOT_DIR")

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

def get_data_dir():
    return get_query_dir('data_root_dir')

def get_dump_dir():
    return get_query_dir('dump_dir')


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
