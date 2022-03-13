import torch
import numpy as np
import utils.geom as utils_geom

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def get_mem_T_ref(B, mem_coord):
    # sometimes we want the mat itself
    # note this is not a rigid transform

    # for interpretability, let's construct this in two steps...
    # okay this code is actually converting the interval
    # into discrete boxes eg. If I want to convert interval
    # -0.5 to 0.5 into 64 parts formula is (x-xmin)/(xmax-xmin) * parts
    # this exact formula is returned from this function.

    # translation
    vox_coord = mem_coord.coord
    MH, MW, MD = mem_coord.proto.shape

    # translation
    center_T_ref = np.eye(4, dtype=np.float32)
    center_T_ref[0,3] = -vox_coord.XMIN
    center_T_ref[1,3] = -vox_coord.YMIN
    center_T_ref[2,3] = -vox_coord.ZMIN

    VOX_SIZE_X = (vox_coord.XMAX-vox_coord.XMIN)/float(MW)
    VOX_SIZE_Y = (vox_coord.YMAX-vox_coord.YMIN)/float(MH)
    VOX_SIZE_Z = (vox_coord.ZMAX-vox_coord.ZMIN)/float(MD)
    
    # scaling
    mem_T_center = np.eye(4, dtype=np.float32)
    mem_T_center[0,0] = 1./VOX_SIZE_X
    mem_T_center[1,1] = 1./VOX_SIZE_Y
    mem_T_center[2,2] = 1./VOX_SIZE_Z
    if mem_coord.correction:
        mem_T_center[0,3] = -0.5
        mem_T_center[1,3] = -0.5
        mem_T_center[2,3] = -0.5

    rt = np.dot(mem_T_center, center_T_ref)

    mem_T_ref = torch.tensor(rt, device=device)
    mem_T_ref = mem_T_ref.unsqueeze(0)
    mem_T_ref = mem_T_ref.repeat(B,1,1)
    return mem_T_ref

def get_ref_T_mem(B, mem_coord):
    mem_T_ref = get_mem_T_ref(B, mem_coord)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = mem_T_ref.inverse()
    return ref_T_mem

def Mem2Ref(xyz_mem, mem_coord):
    # xyz is B x N x 3, in mem coordinates
    # transforms mem coordinates into ref coordinates
    B, N, C = list(xyz_mem.shape)
    #ref_T_mem = tf.tile(mem_coord.cam_T_vox, [B, 1, 1])
    ref_T_mem = get_ref_T_mem(B, mem_coord)
    xyz_ref = utils_geom.apply_4x4(ref_T_mem, xyz_mem)
    return xyz_ref

def Ref2Mem(xyz, mem_coord):
    # xyz is B x N x 3, in ref coordinates
    # transforms camR coordinates into mem coordinates
    # (0, 0, 0) corresponds to (Xmin, Ymin, Zmin)
    B, N, C = list(xyz.shape)
    mem_T_ref = get_mem_T_ref(B, mem_coord)
    xyz = utils_geom.apply_4x4(mem_T_ref, xyz)
    return xyz