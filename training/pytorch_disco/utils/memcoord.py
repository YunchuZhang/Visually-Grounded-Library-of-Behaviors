import utils.coordTcoord

class Coord:
    def __init__(self, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, FLOOR=0, CEIL=-5):
        self.XMIN = XMIN # right (neg is left)
        self.XMAX = XMAX # right
        self.YMIN = YMIN # down (neg is up)
        self.YMAX = YMAX # down
        self.ZMIN = ZMIN
        self.ZMAX = ZMAX
        self.FLOOR = FLOOR # objects don't dip below this
        self.CEIL = CEIL # objects don't rise above his

    def __repr__(self):
        return str([self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX, self.FLOOR, self.CEIL])


class VoxCoord:
    """
    To define a memory, you need to define a coordinate system
    and the size of the memory (proto)
    """
    def __init__(self, coord, proto, correction=False):
        """
        Don't touch the correction. This is for testing file using the discovery repo as ground truth
        """
        self.proto = proto
        self.coord = coord
        self.correction=correction
        self.vox_T_cam = None
        self.cam_T_vox = None

        self.build(self.coord, self.proto)

    def __repr__(self):
        return f'proto: {self.proto} \ncoord: {self.coord}'

    def build(self, coord, proto):
        if self.vox_T_cam == None:
            self.vox_T_cam = utils.coordTcoord.get_mem_T_ref(1, self)
        if self.cam_T_vox == None:
            self.cam_T_vox = self.vox_T_cam.inverse()