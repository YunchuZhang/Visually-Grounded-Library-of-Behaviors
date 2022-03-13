import torch
#import hyperparams as hyp
import numpy as np
import torch.nn.functional as F


import utils.basic as utils_basic
import utils.geom as utils_geom
import utils.samp as utils_samp
import utils.improc as utils_improc

from utils.coordTcoord import Ref2Mem, Mem2Ref, get_ref_T_mem, get_mem_T_ref
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

def get_inbounds(xyz, mem_coord, already_mem=False):
    # xyz is B x N x 3
    vox_coord = mem_coord.coord
    Y, X, Z = mem_coord.proto.shape
    if not already_mem:
        xyz = Ref2Mem(xyz, mem_coord)

    x = xyz[:,:,0]
    y = xyz[:,:,1]
    z = xyz[:,:,2]

    x_valid = (x>-0.5).byte() & (x<float(X-0.5)).byte()
    y_valid = (y>-0.5).byte() & (y<float(Y-0.5)).byte()
    z_valid = (z>-0.5).byte() & (z<float(Z-0.5)).byte()

    inbounds = x_valid & y_valid & z_valid
    return inbounds.bool()

def convert_boxlist_memR_to_camR(boxlist_memR, coords):
    B, N, D = list(boxlist_memR.shape)
    assert(D==9)
    cornerlist_memR_legacy = utils_geom.transform_boxes_to_corners(boxlist_memR)
    ref_T_mem = get_ref_T_mem(B, coords)
    cornerlist_camR_legacy = utils_geom.apply_4x4_to_corners(ref_T_mem, cornerlist_memR_legacy)
    # boxlist_camR = utils_geom.transform_corners_to_boxes(cornerlist_camR_legacy)
    # I want to predict axis aligned boxes as well
    boxlist_camR = utils_geom.convert_corners_to_axis_aligned_boxlist(cornerlist_camR_legacy)
    return boxlist_camR

def get_inbounds_single(xyz, mem_coord, already_mem=False):
    # xyz is N x 3
    xyz = xyz.unsqueeze(0)
    inbounds = get_inbounds(xyz, mem_coord, already_mem=already_mem)
    inbounds = inbounds.squeeze(0)
    return inbounds

def voxelize_xyz(xyz_ref, mem_coord, already_mem=False):
    Y, X, Z = mem_coord.proto.shape
    B, N, D = list(xyz_ref.shape) # batch, num_points, 3
    assert(D==3)
    if already_mem:
        xyz_mem = xyz_ref
    else:
        # no checking of bounds here points which are outside the
        # range of (min, max) are assigned some grid coordinates
        # which are implausible
        xyz_mem = Ref2Mem(xyz_ref, mem_coord)
    vox = get_occupancy(xyz_mem, mem_coord)
    return vox

def get_occupancy_single(xyz, mem_coord):
    # xyz is N x 3 and in mem coords
    # we want to fill a voxel tensor with 1's at these inds
    Y, X, Z = mem_coord.proto.shape
    # (we have a full parallelized version, but fill_ray_single needs this)

    inbounds = get_inbounds_single(xyz, mem_coord, already_mem=True)
    xyz = xyz[inbounds]
    # xyz is N x 3

    # this is more accurate than a cast/floor, but runs into issues when a dim==0
    xyz = torch.round(xyz).int()
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

    vox_inds = utils_basic.sub2ind3D(Z, Y, X, z, y, x)
    vox_inds = vox_inds.flatten().long()
    voxels = torch.zeros(Z*Y*X, dtype=torch.float32)
    voxels[vox_inds] = 1.0
    voxels = voxels.reshape(1, Z, Y, X)
    # 1 x Z x Y x X
    return voxels

def get_occupancy(xyz, mem_coord):
    # xyz is B x N x 3 and in mem coords
    # we want to fill a voxel tensor with 1's at these inds
    B, N, C = list(xyz.shape)
    assert(C==3)
    vox_coord = mem_coord.coord
    Y, X, Z = mem_coord.proto.shape

    # these papers say simple 1/0 occupancy is ok:
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
    # cont fusion says they do 8-neighbor interp
    # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

    # get_inbound function clips the value of xyz which are in mem to lie between (-0.5, Mem_X-0.5)
    # so when int of this will be taken all the values will lie between (0 and Mem_X)
    inbounds = get_inbounds(xyz, mem_coord, already_mem=True)
    x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
    mask = torch.zeros_like(x)
    mask[inbounds] = 1.0

    # set the invalid guys to zero
    # we then need to zero out 0,0,0
    # (this method seems a bit clumsy)
    x = x*mask
    y = y*mask
    z = z*mask

    x = torch.round(x)
    y = torch.round(y)
    z = torch.round(z)
    x = torch.clamp(x, 0, X-1).int()
    y = torch.clamp(y, 0, Y-1).int()
    z = torch.clamp(z, 0, Z-1).int()

    x = x.view(B*N)
    y = y.view(B*N)
    z = z.view(B*N)

    # here all the xyz's which have indices greater than that required by the grid
    # are gone and only the good ones remain
    dim3 = X
    dim2 = X * Y
    dim1 = X * Y * Z

    # base = torch.from_numpy(np.concatenate([np.array([i*dim1]) for i in range(B)]).astype(np.int32))
    base = torch.range(0, B-1, dtype=torch.int32, device=DEVICE)*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

    vox_inds = base + z * dim2 + y * dim3 + x
    voxels = torch.zeros(B*Z*Y*X, device=DEVICE).float()
    voxels[vox_inds.long()] = 1.0
    # zero out the singularity
    voxels[base.long()] = 0.0
    voxels = voxels.reshape(B, 1, Z, Y, X)
    # B x 1 x Z x Y x X
    return voxels

def unproject_rgb_to_mem(rgb_camB, pixB_T_camA, mem_coord, device=None):
    # rgb_camB is B x C x H x W
    # pixB_T_camA is B x 4 x 4 (pix_T_camR)

    # rgb lives in B pixel coords
    # we want everything in A memory coords

    # this puts each C-dim pixel in the rgb_camB
    # along a ray in the voxelgrid
    B, C, H, W = list(rgb_camB.shape)

    Y, X, Z = mem_coord.proto.shape

    xyz_memA = utils_basic.gridcloud3D(B, Z, Y, X, norm=False, device=device)
    # grid_z, grid_y, grid_x = meshgrid3D(B, Z, Y, X)
    # # these are B x Z x Y x X
    # # these represent the mem grid coordinates

    # # we need to convert these to pixel coordinates
    # x = torch.reshape(grid_x, [B, -1])
    # y = torch.reshape(grid_y, [B, -1])
    # z = torch.reshape(grid_z, [B, -1])
    # # these are B x N
    # xyz_mem = torch.stack([x, y, z], dim=2)

    # not specifically related to Ref, I am just
    # converting grid to points here, irrespective
    # of the which cam it is associated to.
    xyz_camA = Mem2Ref(xyz_memA, mem_coord)

    xyz_pixB = utils_geom.apply_4x4(pixB_T_camA, xyz_camA)
    # this is just getting the z coordinate to divide x/Z, y/Z
    normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
    EPS=1e-6
    xy_pixB = xyz_pixB[:,:,:2]/(EPS+normalizer)
    # this is B x N x 2
    # this is the (floating point) pixel coordinate of each voxel
    x_pixB, y_pixB = xy_pixB[:,:,0], xy_pixB[:,:,1]
    # these are B x N

    if (0):
        # handwritten version
        values = torch.zeros([B, C, Z*Y*X], dtype=torch.float32)
        for b in range(B):
            values[b] = utils_samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
    else:
        # native pytorch version, this makes the pixel between -1 to 1
        y_pixB, x_pixB = utils_basic.normalize_grid2D(y_pixB, x_pixB, H, W)
        # since we want a 3d output, we need 5d tensors
        z_pixB = torch.zeros_like(x_pixB)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
        rgb_camB = rgb_camB.unsqueeze(2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_camB, xyz_pixB, mode='nearest')

    values = torch.reshape(values, (B, C, Z, Y, X))
    return values

def apply_pixX_T_memR_to_voxR(pix_T_camX, camX_T_camR, mem_coord, voxR, D, H, W):
    # mats are B x 4 x 4
    # voxR is B x C x Z x Y x X
    # H, W, D indicates how big to make the output
    # returns B x C x D x H x W
    vox_coord = mem_coord.coord
    ZMIN = vox_coord.ZMIN
    ZMAX = vox_coord.ZMAX

    B, C, Z, Y, X = list(voxR.shape)
    z_near = ZMIN
    z_far = ZMAX

    grid_z = torch.linspace(z_near, z_far, steps=D, dtype=torch.float32, device=DEVICE)
    grid_z = torch.reshape(grid_z, [1, 1, D, 1, 1])
    grid_z = grid_z.repeat([B, 1, 1, H, W])
    grid_z = torch.reshape(grid_z, [B*D, 1, H, W])

    pix_T_camX__ = torch.unsqueeze(pix_T_camX, axis=1).repeat([1, D, 1, 1])
    pix_T_camX = torch.reshape(pix_T_camX__, [B*D, 4, 4])
    xyz_camX = utils_geom.depth2pointcloud(grid_z, pix_T_camX)

    camR_T_camX = utils_geom.safe_inverse(camX_T_camR)
    camR_T_camX_ = torch.unsqueeze(camR_T_camX, dim=1).repeat([1, D, 1, 1])
    camR_T_camX = torch.reshape(camR_T_camX_, [B*D, 4, 4])

    mem_T_cam = get_mem_T_ref(B*D, mem_coord)
    memR_T_camX = utils_basic.matmul2(mem_T_cam, camR_T_camX)

    xyz_memR = utils_geom.apply_4x4(memR_T_camX, xyz_camX)
    xyz_memR = torch.reshape(xyz_memR, [B, D*H*W, 3])

    samp = utils_samp.sample3D(voxR, xyz_memR, D, H, W)
    # samp is B x H x W x D x C
    return samp

def assemble_static_seq(feats, ref0_T_refXs):
    # feats is B x S x C x Y x X x Z
    # it is in mem coords

    # ref0_T_refXs is B x S x 4 x 4
    # it tells us how to warp the static scene

    # ref0 represents a reference frame, not necessarily frame0
    # refXs represents the frames where feats were observed


    B, S, C, Z, Y, X = list(feats.shape)

    # each feat is in its own little coord system
    # we need to get from 0 coords to these coords
    # and sample

    # we want to sample for each location in the bird grid
    # xyz_mem = gridcloud3D(B, Z, Y, X)
    grid_y, grid_x, grid_z = meshgrid3D(B, Z, Y, X)
    # these are B x BY x BX x BZ
    # these represent the mem grid coordinates

    # we need to convert these to pixel coordinates
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N

    xyz_mem = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    xyz_ref = Mem2Ref(xyz_mem, Z, Y, X)
    # this is B x N x 3
    xyz_refs = xyz_ref.unsqueeze(1).repeat(1,S,1,1)
    # this is B x S x N x 3
    xyz_refs_ = torch.reshape(xyz_refs, (B*S, Y*X*Z, 3))

    feats_ = torch.reshape(feats, (B*S, C, Z, Y, X))

    ref0_T_refXs_ = torch.reshape(ref0_T_refXs, (B*S, 4, 4))
    refXs_T_ref0_ = utils_geom.safe_inverse(ref0_T_refXs_)

    xyz_refXs_ = utils_geom.apply_4x4(refXs_T_ref0_, xyz_refs_)
    xyz_memXs_ = Ref2Mem(xyz_refXs_, Z, Y, X)
    feats_, _ = utils_samp.resample3D(feats_, xyz_memXs_)
    feats = torch.reshape(feats_, (B, S, C, Z, Y, X))
    return feats

def resample_to_target_views(occRs, camRs_T_camPs):
    # resample to the target view

    # occRs is B x S x Y x X x Z x 1
    # camRs_T_camPs is B x S x 4 x 4

    B, S, _, Z, Y, X = list(occRs.shape)

    # we want to construct a mat memR_T_memP

    cam_T_mem = get_ref_T_mem(B, Z, Y, X)
    mem_T_cam = get_mem_T_ref(B, Z, Y, X)
    cams_T_mems = cam_T_mem.unsqueeze(1).repeat(1, S, 1, 1)
    mems_T_cams = mem_T_cam.unsqueeze(1).repeat(1, S, 1, 1)

    cams_T_mems = torch.reshape(cams_T_mems, (B*S, 4, 4))
    mems_T_cams = torch.reshape(mems_T_cams, (B*S, 4, 4))
    camRs_T_camPs = torch.reshape(camRs_T_camPs, (B*S, 4, 4))

    memRs_T_memPs = torch.matmul(torch.matmul(mems_T_cams, camRs_T_camPs), cams_T_mems)
    memRs_T_memPs = torch.reshape(memRs_T_memPs, (B, S, 4, 4))

    occRs, valid = resample_to_view(occRs, memRs_T_memPs, multi=True)
    return occRs, valid

def resample_to_target_view(occRs, camR_T_camP):
    B, S, Z, Y, X, _ = list(occRs.shape)
    cam_T_mem = get_ref_T_mem(B, Z, Y, X)
    mem_T_cam = get_mem_T_ref(B, Z, Y, X)
    memR_T_memP = torch.matmul(torch.matmul(mem_T_cam, camR_T_camP), cam_T_mem)
    occRs, valid = resample_to_view(occRs, memR_T_memP, multi=False)
    return occRs, valid

def resample_to_view(feats, new_T_old, multi=False):
    # feats is B x S x c x Y x X x Z
    # it represents some scene features in reference/canonical coordinates
    # we want to go from these coords to some target coords

    # new_T_old is B x 4 x 4
    # it represents a transformation between two "mem" systems
    # or if multi=True, it's B x S x 4 x 4

    B, S, C, Z, Y, X = list(feats.shape)

    # we want to sample for each location in the bird grid
    # xyz_mem = gridcloud3D(B, Z, Y, X)
    grid_y, grid_x, grid_z = meshgrid3D(B, Z, Y, X)
    # these are B x BY x BX x BZ
    # these represent the mem grid coordinates

    # we need to convert these to pixel coordinates
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N

    xyz_mem = torch.stack([x, y, z], dim=2)
    # this is B x N x 3

    xyz_mems = xyz_mem.unsqueeze(1).repeat(1, S, 1, 1)
    # this is B x S x N x 3

    xyz_mems_ = xyz_mems.view(B*S, Y*X*Z, 3)

    feats_ = feats.view(B*S, C, Z, Y, X)

    if multi:
        new_T_olds = new_T_old.clone()
    else:
        new_T_olds = new_T_old.unsqueeze(1).repeat(1, S, 1, 1)
    new_T_olds_ = new_T_olds.view(B*S, 4, 4)

    xyz_new_ = utils_geom.apply_4x4(new_T_olds_, xyz_mems_)
    # we want each voxel to replace its value
    # with whatever is at these new coordinates

    # i.e., we are back-warping from the "new" coords

    feats_, valid_ = utils_samp.resample3D(feats_, xyz_new_)
    feats = feats_.view(B, S, C, Z, Y, X)
    valid = valid_.view(B, S, 1, Z, Y, X)
    return feats, valid

def convert_xyz_to_visibility(xyz, mem_coord):
    # xyz is in camera coordinates
    # proto shows the size of the birdgrid
    B, N, C = list(xyz.shape)
    Y, X, Z = mem_coord.proto.shape
    assert(C==3)
    voxels = torch.zeros(B, 1, Z, Y, X, dtype=torch.float32, device=DEVICE)
    for b in range(B):
        voxels[b,0] = fill_ray_single(xyz[b], mem_coord)
    return voxels

def fill_ray_single(xyz, mem_coord):
    # xyz is N x 3, and in bird coords
    # we want to fill a voxel tensor with 1's at these inds,
    # and also at any ind along the ray before it
    Y, X, Z = mem_coord.proto.shape
    xyz = torch.reshape(xyz, (-1, 3))
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    # these are N

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)

    # get the hypotenuses
    u = torch.sqrt(x**2+z**2) # flat to ground
    v = torch.sqrt(x**2+y**2+z**2)
    w = torch.sqrt(x**2+y**2)

    # the ray is along the v line
    # we want to find xyz locations along this line

    # get the angles
    EPS=1e-6
    sin_theta = y/(EPS + v) # soh
    cos_theta = u/(EPS + v) # cah
    sin_alpha = z/(EPS + u) # soh
    cos_alpha = x/(EPS + u) # cah

    samps = int(np.sqrt(Y**2 + Z**2))
    # for each proportional distance in [0.0, 1.0], generate a new hypotenuse
    dists = torch.linspace(0.0, 1.0, samps, device=DEVICE)
    dists = torch.reshape(dists, (1, samps))
    v_ = dists * v.repeat(1, samps)

    # now, for each of these v_, we want to generate the xyz
    y_ = sin_theta*v_
    u_ = torch.abs(cos_theta*v_)
    z_ = sin_alpha*u_
    x_ = cos_alpha*u_
    # these are the ref coordinates we want to fill
    x = x_.flatten()
    y = y_.flatten()
    z = z_.flatten()

    xyz = torch.stack([x,y,z], dim=1).unsqueeze(0)
    xyz = Ref2Mem(xyz, mem_coord)
    xyz = torch.squeeze(xyz, dim=0)
    # these are the mem coordinates we want to fill

    return get_occupancy_single(xyz, mem_coord)

def get_freespace(xyz, occ, mem_coord):
    # xyz is B x N x 3
    # occ is B x H x W x D x 1
    B, C, Z, Y, X = list(occ.shape)
    assert(C==1)
    # check the resulting mem_coord is of the same size as current occ
    assert(Y == mem_coord.proto.shape[0])
    assert(X == mem_coord.proto.shape[1])
    assert(Z == mem_coord.proto.shape[2])

    vis = convert_xyz_to_visibility(xyz, mem_coord)
    # visible space is all free unless it's occupied
    free = (1.0-(occ>0.0).float())*vis
    return free


def apply_4x4_to_vox(B_T_A, feat_A, mem_coord_As=None, mem_coord_Bs = None, already_mem=False, binary_feat=False, rigid=True):
    # B_T_A is B x 4 x 4
    # if already_mem=False, it is a transformation between cam systems
    # if already_mem=True, it is a transformation between mem systems

    # feat_A is B x C x Z x Y x X
    # it represents some scene features in reference/canonical coordinates
    # we want to go from these coords to some target coords

    # since this is a backwarp,
    # the question to ask is:
    # "WHERE in the tensor do you want to sample,
    # to replace each voxel's current value?"

    # the inverse of B_T_A represents this "where";
    # it transforms each coordinate in B
    # to the location we want to sample in A

    B, C, Z, Y, X = list(feat_A.shape)

    # we have B_T_A in input, since this follows the other utils_geom.apply_4x4
    # for an apply_4x4 func, but really we need A_T_B
    if rigid:
        A_T_B = utils_geom.safe_inverse(B_T_A)
    else:
        # this op is slower but more powerful
        A_T_B = B_T_A.inverse()


    if not already_mem:
        cam_T_mem =  mem_coord_Bs.cam_T_vox.repeat(B, 1, 1)
        mem_T_cam = mem_coord_As.vox_T_cam.repeat(B, 1, 1)
        A_T_B = utils_basic.matmul3(mem_T_cam, A_T_B, cam_T_mem)

    # we want to sample for each location in the bird grid
    xyz_B = utils_basic.gridcloud3D(B, Z, Y, X)
    # this is B x N x 3

    # transform
    xyz_A = utils_geom.apply_4x4(A_T_B, xyz_B)
    # we want each voxel to take its value
    # from whatever is at these A coordinates
    # i.e., we are back-warping from the "A" coords

    # feat_B = F.grid_sample(feat_A, normalize_grid(xyz_A, Z, Y, X))
    feat_B = utils_samp.resample3D(feat_A, xyz_A, binary_feat=binary_feat)

    # feat_B, valid = utils_samp.resample3D(feat_A, xyz_A, binary_feat=binary_feat)
    # return feat_B, valid
    return feat_B

def apply_4x4_to_voxs(Bs_T_As, feat_As, mem_coord_As=None, mem_coord_Bs = None, already_mem=False, binary_feat=False):
    # plural wrapper for apply_4x4_to_vox

    B, S, C, Z, Y, X = list(feat_As.shape)

    # utils for packing/unpacking along seq dim
    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)

    Bs_T_As_ = __p(Bs_T_As)
    feat_As_ = __p(feat_As)
    feat_Bs_ = apply_4x4_to_vox(Bs_T_As_, feat_As_, mem_coord_As=mem_coord_As, mem_coord_Bs = mem_coord_Bs, already_mem=already_mem, binary_feat=binary_feat)
    feat_Bs = __u(feat_Bs_)
    return feat_Bs

def prep_occs_supervision(xyz_camXs,
                          occRs,
                          occXs,
                          camRs_T_camXs,
                          memcoord_camRs,
                          memcoord_camXs,
                          agg=False):
                          # writer):
                          # global_step,
                          # vis=True,
                          # agg=False):
    B, S, C, Z, Y, X = list(occXs.size())
    assert(C==1)
    assert(Y == memcoord_camXs.proto.shape[0])
    assert(X == memcoord_camXs.proto.shape[1])
    assert(Z == memcoord_camXs.proto.shape[2])


    # and we want the free space to have the same size
    TH, TW, TD = memcoord_camRs.proto.shape
    assert(Y== TH)
    assert(X == TW)
    assert(Z == TD)

    xyz_camXs_ = xyz_camXs.view(B*S, -1, 3)
    occXs_ = occXs.view(B*S, 1, Z, Y, X)
    freeXs_ = get_freespace(xyz_camXs_, occXs_, memcoord_camXs)
    # this is B*S x 1 x Z x Y x X

    freeXs = freeXs_.view(B, S, 1, Z, Y, X)
    freeRs = apply_4x4_to_voxs(camRs_T_camXs, freeXs, mem_coord_As=memcoord_camXs, mem_coord_Bs = memcoord_camRs)
    # this is B x S x 1 x Z x Y x X

    if agg:
        # we should only agg if we are in STATIC mode (time frozen)
        # and note we tile S-1 of them, since static mode skips the ref frame
        freeR = torch.max(freeRs, dim=1)[0]
        occR = torch.max(occRs, dim=1)[0]
        # # these are B x 1 x Z x Y x X
        # freeRs = torch.unsqueeze(freeR, dim=1)#.repeat(1, S, 1, 1, 1, 1)
        # occRs = torch.unsqueeze(occR, dim=1)#.repeat(1, S, 1, 1, 1, 1)
        occR = (occR>0.5).float()
        freeR = (freeR>0.5).float()
        return occR, freeR, freeXs
    else:
        occRs = (occRs>0.5).float()
        freeRs = (freeRs>0.5).float()
        return occRs, freeRs, freeXs

def assemble_padded_obj_masklist(lrtlist, scorelist, Z, Y, X, coeff=1.0):
    # compute a binary mask in 3D for each object
    # we use this when computing the center-surround objectness score
    # lrtlist is B x N x 19
    # scorelist is B x N

    # returns masklist shaped B x N x 1 x Z x Y x Z

    B, N, D = list(lrtlist.shape)
    assert(D==19)
    masks = torch.zeros(B, N, Z, Y, X)

    lenlist, ref_T_objlist = utils_geom.split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # ref_T_objlist is B x N x 4 x 4

    lenlist_ = lenlist.reshape(B*N, 3)
    ref_T_objlist_ = ref_T_objlist.reshape(B*N, 4, 4)
    obj_T_reflist_ = utils_geom.safe_inverse(ref_T_objlist_)

    # we want a value for each location in the mem grid
    xyz_mem_ = utils_basic.gridcloud3D(B*N, Z, Y, X)
    # this is B*N x V x 3, where V = Z*Y*X
    xyz_ref_ = Mem2Ref(xyz_mem_, Z, Y, X)
    # this is B*N x V x 3

    lx, ly, lz = torch.unbind(lenlist_, dim=1)
    # these are B*N

    # ref_T_obj = convert_box_to_ref_T_obj(boxes3D)
    # obj_T_ref = ref_T_obj.inverse()

    xyz_obj_ = utils_geom.apply_4x4(obj_T_reflist_, xyz_ref_)
    x, y, z = torch.unbind(xyz_obj_, dim=2)
    # these are B*N x V

    lx = lx.unsqueeze(1)*coeff
    ly = ly.unsqueeze(1)*coeff
    lz = lz.unsqueeze(1)*coeff
    # these are B*N x 1

    x_valid = (x > -lx/2.0).byte() & (x < lx/2.0).byte()
    y_valid = (y > -ly/2.0).byte() & (y < ly/2.0).byte()
    z_valid = (z > -lz/2.0).byte() & (z < lz/2.0).byte()
    inbounds = x_valid.byte() & y_valid.byte() & z_valid.byte()
    masklist = inbounds.float()
    # print(masklist.shape)
    masklist = masklist.reshape(B, N, 1, Z, Y, X)
    # print(masklist.shape)
    # print(scorelist.shape)
    masklist = masklist*scorelist.view(B, N, 1, 1, 1, 1)
    return masklist

def get_zoom_T_ref(lrt, ZZ, ZY, ZX):
    # lrt is B x 19
    B, E = list(lrt.shape)
    assert(E==19)
    lens, ref_T_obj = utils_geom.split_lrt(lrt)
    lx, ly, lz = lens.unbind(1)
    # print('lx, ly, lz')
    # print(lx)
    # print(ly)
    # print(lz)

    obj_T_ref = utils_geom.safe_inverse(ref_T_obj)
    # this is B x 4 x 4

    # print('ok, got obj_T_ref:')
    # print(obj_T_ref)

    # we want a tiny bit of padding
    # additive helps avoid nans with invalid objects
    # mult helps expand big objects
    lx = lx + 0.05
    ly = ly + 0.05
    lz = lz + 0.05
    # lx *= 1.1
    # ly *= 1.1
    # lz *= 1.1

    # translation
    center_T_obj_r = utils_geom.eye_3x3(B)
    center_T_obj_t = torch.stack([lx/2., ly/2., lz/2.], dim=1)
    # print('merging these:')
    # print(center_T_obj_r.shape)
    # print(center_T_obj_t.shape)
    center_T_obj = utils_geom.merge_rt(center_T_obj_r, center_T_obj_t)

    # print('ok, got center_T_obj:')
    # print(center_T_obj)

    # scaling
    Z_VOX_SIZE_X = (lx)/float(ZX)
    Z_VOX_SIZE_Y = (ly)/float(ZY)
    Z_VOX_SIZE_Z = (lz)/float(ZZ)
    diag = torch.stack([1./Z_VOX_SIZE_X,
                        1./Z_VOX_SIZE_Y,
                        1./Z_VOX_SIZE_Z,
                        torch.ones([B])], axis=1).view(B, 4)
    # print('diag:')
    # print(diag)
    zoom_T_center = torch.diagflat(diag[0]).unsqueeze(0).repeat(B, 1, 1)
    # print('ok, got zoom_T_center:')
    # print(zoom_T_center)

    # compose these, this would convert continuous coordinates to grid coordinates
    zoom_T_obj = utils_basic.matmul2(zoom_T_center, center_T_obj)

    # print('ok, got zoom_T_obj:')
    # print(zoom_T_obj)

    # obj_T_ref takes a point in camera coordinates to box coordinates
    # zoom_T_obj takes coordinate in box frame to grid coordinates
    zoom_T_ref = utils_basic.matmul2(zoom_T_obj, obj_T_ref)

    # print('ok, got zoom_T_ref:')
    # print(zoom_T_ref)

    return zoom_T_ref

def get_ref_T_zoom(lrt, ZY, ZX, ZZ):
    zoom_T_ref = get_zoom_T_ref(lrt, ZY, ZX, ZZ)
    # note safe_inverse is inapplicable here,
    # since the transform is composition of non-rigid
    # and rigid transformation
    # this converts grid coordinates to box to cam coordinates
    ref_T_zoom = zoom_T_ref.inverse()
    return ref_T_zoom

def Ref2Zoom(xyz_ref, lrt_ref, ZY, ZX, ZZ):
    # xyz_ref is B x N x 3, in ref coordinates
    # lrt_ref is B x 9, specifying the box in ref coordinates
    # this transforms ref coordinates into zoom coordinates
    B, N, _ = list(xyz_ref.shape)
    zoom_T_ref = get_zoom_T_ref(lrt_ref, ZY, ZX, ZZ)
    xyz_zoom = utils_geom.apply_4x4(zoom_T_ref, xyz_ref)
    return xyz_zoom

def Zoom2Ref(xyz_zoom, lrt_ref, ZY, ZX, ZZ, sensor_camR_T_camXs=None):
    # xyz_zoom is B x N x 3, in zoom coordinates
    # lrt_ref is B x 9, converts from box to cam coordinates
    # sensor_camR_T_camXs standard transformation matrix to
    # convert from cam coords to ref_coords.
    B, N, _  = list(xyz_zoom.shape)
    ref_T_zoom = get_ref_T_zoom(lrt_ref, ZY, ZX, ZZ)
    ref_T_zoom = ref_T_zoom.to(xyz_zoom.device)
    # the zero zero zero of xyz_zoom should be mapped to (0, 0, 0.07)
    # and it does I checked it
    if sensor_camR_T_camXs is not None:
        # this takes from grid_coordinates to box_coordinates to cam_coordinates
        # to ref_coordinates
        ref_T_zoom = utils_basic.matmul2(sensor_camR_T_camXs, ref_T_zoom)
    # remember this are coordinates in ref_cam and not in memory
    xyz_ref = utils_geom.apply_4x4(ref_T_zoom, xyz_zoom)
    return xyz_ref

def crop_zoom_from_mem(mem, mem_coord, lrt, Z2, Y2, X2, sensor_camR_T_camXs=None):
    # mem is B x C x Z x Y x X
    # lrt is B x 9 it takes me from object to cam coords
    # sensor_camR_T_camXs takes me from cam coords to ref coords

    B, C, Z, Y, X = list(mem.shape)  ## 2, 512, 32, 32, 32
    memY, memX, memZ = mem_coord.proto.shape

    assert(Z == memZ)
    assert(Y == memY)
    assert(X == memX)
    B2, E = list(lrt.shape)  ## 2, 19
    # I do not particularly like this inclusion of if statement
    if sensor_camR_T_camXs is not None:
        B3, _, _ = list(sensor_camR_T_camXs.shape)

    assert(E==19)
    assert(B==B2)
    if sensor_camR_T_camXs is not None:
        assert(B==B3)

    # this puts each C-dim pixel in the image
    # along a ray in the zoomed voxelgrid just
    # the grid it has no coordinate system attached to it

    xyz_zoom = utils_basic.gridcloud3D(B, Z2, Y2, X2, norm=False)
    # these represent the zoom grid coordinates
    # we need to convert these to mem coordinates
    xyz_ref = Zoom2Ref(xyz_zoom, lrt, Z2, Y2, X2, sensor_camR_T_camXs)
    xyz_mem = Ref2Mem(xyz_ref, mem_coord)  # now it is in grid coordinates of mem

    # this is just like the grid sample
    zoom = utils_samp.sample3D(mem, xyz_mem, Z2, Y2, X2)
    zoom = torch.reshape(zoom, [B, C, Z2, Y2, X2])
    return zoom
