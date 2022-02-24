from pcp_utils.utils import Config
import pcp_utils
import os


class MeshObject:
    class Config(Config):
        obj_xml_file = ""
        class_id = None
        scale = 1.0
        mass = 0.01
        euler_xyz = [0, 0, 0]
        cluster_id = ''
        success_rates_over_class = ""

    def __init__(self, config:Config, name):
        self.config = config
        self.name = name
        object_xml_dir = pcp_utils.utils.get_object_xml_dir()
        self.obj_xml_file = os.path.join(object_xml_dir, config.obj_xml_file)

        if not os.path.exists(self.obj_xml_file):
            print(f"cannot find the file: {self.obj_xml_file}")
            raise ValueError

        self.class_id = config.class_id
        self.scale = config.scale
        self.mass = config.mass

        if isinstance(config.euler_xyz, str):
            config.euler_xyz = [float(item) for item in config.euler_xyz.split(" ")]
        self.euler = config.euler_xyz
        self.cluster_id = config.cluster_id
        self.success_rates_over_class = config.success_rates_over_class

    def step(self, obs):
        raise NotImplementedError("Must be implemented in subclass.")





