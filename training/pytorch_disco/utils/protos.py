class VoxProto:
    def __init__(self, shape):
        self.dim = len(shape)
        self.shape = shape
        self.H = shape[0]
        self.W = shape[1]
        if len(shape) >= 3:
            self.D = shape[2]


    def __repr__(self):
        return str(self.shape)