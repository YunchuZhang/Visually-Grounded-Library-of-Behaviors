import numpy as np
import sys
import open3d as o3d
import tensorflow as tf
import os

import pcp_utils
from pcp_utils.utils import Config


graspnet_path = pcp_utils.utils.get_6dof_graspnet_dir()
sys.path.append(graspnet_path)

import grasp_estimator
import tf_utils

from grasp_data_reader import regularize_pc_point_count

tf.set_random_seed(0)

def _regularize_pc_point_count(pcs, npoints):
    return regularize_pc_point_count(pcs, npoints)



class GraspSampler:

    class Config(Config):
        # path for checkpoints
        #vae_checkpoint_folder = f'{graspnet_path}/checkpoints/npoints_1024_train_evaluator_0_allowed_categories__ngpus_1_'
        #vae_checkpoint_folder = f'{graspnet_path}/log/vae_finetune_20_gan1_model_gan1/'
        #vae_checkpoint_folder = f'{graspnet_path}/log/vae_finetune_mug1_50_fv_na0_gan0_all_train_small_var10_lr5/'
        vae_checkpoint_folder = f'{graspnet_path}/checkpoints/latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_'
        evaluator_checkpoint_folder = f'{graspnet_path}/checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_/'
        #evaluator_checkpoint_folder = f'{graspnet_path}/log/evaluator_finetune_mug1_50_fv_all_train_small_var10'
        #evaluator_checkpoint_folder = f'{graspnet_path}/log/evaluator_finetune_20_gan1_40k'
        gradient_based_refinement = False
        grasp_conf_threshold = 0.8
        cut_off_points = [0.3, 0.5, 0.5] #should remove

        # camera params
        camera_img_height = 128
        camera_img_width = 128
        camera_radius = 0.3
        camera_fov_y = 45
        camera_pitch = [40, 41, 2]
        camera_yaw = [0, 350, 36]
        camera_yaw_list = None #[0, 60, 300]
        camera_save_image = False
        camera_recon_scene = True
        camera_lookat_pos = [1.3, 0.75, 0.4]

        table_top = [1.3, 0.75, 0.4]
        table_T_camR = [0, 0, 0]
        cut_off_points = [0.3, 0.5, 0.5]  # for cropping pointcloud

        data_collection_mode = False
        data_collection_from_trained_model = False
        save_data_name = None
        fix_view = False

    def __init__(self, config:Config):
        self.config = config
        self.vae_checkpoint_folder = config.vae_checkpoint_folder
        self.evaluator_checkpoint_folder = config.evaluator_checkpoint_folder

        self.gradient_based_refinement = False
        self.grasp_conf_threshold = 0.8
        self.cut_off_points = config.cut_off_points
        self.output_grasps_dir = "vae_generated_grasps"
        self.fix_view = config.fix_view

    
        ##### Prepare the 6dof graspnet network for forward pass ######
        cfg = grasp_estimator.joint_config(
            self.vae_checkpoint_folder,
            self.evaluator_checkpoint_folder,
        )
        cfg['threshold'] = self.grasp_conf_threshold
        cfg['sample_based_improvement'] = 1 - int(self.gradient_based_refinement)
        cfg['num_refine_steps'] = 10 if self.gradient_based_refinement else 20


        if self.config.data_collection_mode:
            if not self.config.data_collection_from_trained_model:
                cfg["use_geometry_sampling"] = True
                cfg['num_refine_steps'] = 0
                cfg['grasp_selection_mode'] = "all"


        #cfg['num_refine_steps'] = 0
        self.num_refine_steps = cfg['num_refine_steps']
        self.estimator = grasp_estimator.GraspEstimator(cfg)
        self.sess = tf.Session()
        self.estimator.build_network()
        self.estimator.load_weights(self.sess)



        if not os.path.exists(self.output_grasps_dir):
            os.makedirs(self.output_grasps_dir)

        # set camera for this:
        self.camera_positions, self.camera_quats = pcp_utils.cameras.generate_new_cameras_hemisphere(radius=self.config.camera_radius,
            lookat_point=self.config.camera_lookat_pos, pitch=self.config.camera_pitch, yaw=self.config.camera_yaw, yaw_list=self.config.camera_yaw_list)
        self.n_cams = len(self.camera_positions)

        # i don't think we need this
        mujoco_T_adam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        origin_T_camR_xpos = np.array(config.table_top, np.float32) + np.array(config.table_T_camR, np.float)

        origin_T_adam = np.zeros((4,4), dtype=np.float32)
        origin_T_adam[:3, :3] = mujoco_T_adam
        origin_T_adam[:3, 3] = origin_T_camR_xpos
        origin_T_adam[3,3] = 1
        self.origin_T_adam = origin_T_adam
        self.adam_T_origin = np.linalg.inv(self.origin_T_adam)
    
        gripper_pc = np.squeeze(tf_utils.get_control_point_tensor(1, False), 0)

        gripper_pc[2, 2] = 0.059
        gripper_pc[3, 2] = 0.059
        mid_point = 0.5*(gripper_pc[2, :] + gripper_pc[3, :])

        # modified grasps
        modified_gripper_pc = []
        modified_gripper_pc.append(np.zeros((3,), np.float32))
        modified_gripper_pc.append(mid_point)
        modified_gripper_pc.append(gripper_pc[2])
        modified_gripper_pc.append(gripper_pc[4])
        modified_gripper_pc.append(gripper_pc[2])
        modified_gripper_pc.append(gripper_pc[3])
        modified_gripper_pc.append(gripper_pc[5])


        self.gripper_pc_ori = np.asarray(modified_gripper_pc)


    def convert_to_adam(self, images):
        """
        This is copied from policy_compression_with_init_policies_3dtensor.py
        TODO: @Fish: would it be better to move this func to utils
        NOTE: Modifies the images dict to add a new key
        """
        origin_T_camXs = images['origin_T_camXs']
        camR_T_camXs = []
        for origin_T_camX in origin_T_camXs:
            camR_T_camX = np.dot(self.adam_T_origin, origin_T_camX)
            camR_T_camXs.append(camR_T_camX)

        camR_T_camXs = np.stack(camR_T_camXs, axis=0)
        images['camR_T_camXs'] = camR_T_camXs
        return images

    def convert_grasp_to_mujoco(self, grasp_pos, grasp_orn):

        grasp_pos_addones = np.ones((4, 1), np.float32)
        grasp_pos_addones[:3, 0] = grasp_pos
        grasp_pos = np.matmul(self.origin_T_adam, grasp_pos_addones)[:3, 0]
        grasp_orn_addones = np.eye(4)
        grasp_orn_addones[:3, :3] = grasp_orn
        grasp_orn = np.matmul(self.origin_T_adam, grasp_orn_addones)
        grasp_orn = grasp_orn[:3, :3]

        return grasp_pos, grasp_orn

    def render_images(self, env):
        """
        This is copied from policy_compression_with_init_policies_3dtensor.py
        TODO: @Fish: would it be better to move this func to utils
        """
        rgbs = []
        depths = []
        pix_T_camXs = []
        origin_T_camXs = []
        for cam_id in range(self.n_cams):
            # need to reset everytime you want to take the picture: the camera has mass and it will fall during execution
            env.set_camera(self.camera_positions[cam_id, :], self.camera_quats[cam_id, :], camera_name= f"ext_camera_0")
            rgb, depth = env.render_from_camera(self.config.camera_img_height, self.config.camera_img_width, camera_name=f"ext_camera_0")

            # need to convert depth to real numbers
            pix_T_camX = pcp_utils.cameras.get_intrinsics(self.config.camera_fov_y, self.config.camera_img_width, self.config.camera_img_height)
            origin_T_camX = pcp_utils.cameras.gymenv_get_extrinsics(env, f'ext_camera_0')

            rgbs.append(rgb)
            depths.append(depth)
            pix_T_camXs.append(pix_T_camX)
            origin_T_camXs.append(origin_T_camX)
    
        images = dict()
        images['rgb_camXs'] = np.stack(rgbs, axis=0)
        images['depth_camXs'] = np.stack(depths, axis=0)
        images['pix_T_camXs'] = np.stack(pix_T_camXs, axis=0)
        images['origin_T_camXs'] = np.stack(origin_T_camXs, axis=0)
        return images
   

    def extract_object_pcd(self, obj_and_table_pts, save=False, obj_name=None):
        pcd = o3d.geometry.PointCloud()

        obj_and_table_pts[:,2] > self.config.table_top[2]

        pcd.points = o3d.utility.Vector3dVector(obj_and_table_pts)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)


        obj_pts = np.asarray(outlier_cloud.points)

        if save:
            save_dir = "cropped_pc"
            self.save_pcd(obj_pts, save_dir, obj_name)
        return obj_pts
        
    @staticmethod
    def save_pcd(pts, save_dir=None, obj_name=None, color=None):
        if save_dir is None:
            save_dir = "tmp"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # if the name is none then assign one
        if obj_name is None:
            obj_name = "recon_pts.npy"
        else:
            obj_name = f"{obj_name}.npy"

        if color is not None:
            save_dict = dict(pts = pts, color = color)
            np.save(os.path.join(save_dir, obj_name), save_dict)
        else:
            np.save(os.path.join(save_dir, obj_name), pts)

    def preprocess_points_for_6dgrasp(self, env, obj_info, images, save=False, save_dir=None, obj_name=None, canonicalize_pcs=False):
        rgbs = images['rgb_camXs']
        depths = images['depth_camXs']
        pix_T_camXs = images['pix_T_camXs']
        camR_T_camXs = images['camR_T_camXs']
        
        # unproject to get the pointcloud
        _, xyz_camRs, _ = pcp_utils.np_vis.unproject_depth(
            depths, pix_T_camXs, camR_T_camXs,
            camR_T_origin = None, clip_radius=1.0,
            do_vis=False
        )

        #        eyes = np.tile(np.expand_dims(np.eye(4), 0), [10, 1, 1])
        #
        #        _, xyz_camXs, _ = pcp_utils.np_vis.unproject_depth(
        #            depths, pix_T_camXs, eyes,
        #            camR_T_origin = None, clip_radius=1.0,
        #            do_vis=False
        #        )
        #        xyz_camXs = np.stack(xyz_camXs, axis=0)
        #        xyz_camXs = xyz_camXs.reshape(-1, 3)
        #        inliers_x = np.abs(xyz_camXs[:, 0]) <= 10
        #        inliers_y = np.abs(xyz_camXs[:, 1]) <= 10
        #        inliers_z = np.abs(xyz_camXs[:, 2]) <= 10
        #
        #        selection = inliers_x & inliers_y & inliers_z
        #
        #        pc2 = xyz_camXs.copy()
        #        xyz_camXs = pc2[selection,:]

        #import ipdb; ipdb.set_trace()
        
        # NOTE: I am using all the views here, TODO: make it use one view
        xyz_camRs = np.stack(xyz_camRs, axis=0)
        xyz_camRs = xyz_camRs.reshape(-1, 3)

        # compute the inliers, I am using priviledged info about table size

        # not good. should use object bounding box



        inliers_x = np.abs(xyz_camRs[:, 0]) <= self.cut_off_points[0]
        inliers_y = np.abs(xyz_camRs[:, 1]) <= self.cut_off_points[1]
        inliers_z = np.abs(xyz_camRs[:, 2]) <= self.cut_off_points[2]

        selection = inliers_x & inliers_y & inliers_z
        
        pc = xyz_camRs.copy()
        pc = pc[selection, :]


        #        import open3d as o3d
        #
        #
        #        def make_frame():
        #            return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #        def make_pcd(pts):
        #            assert pts.shape[1] == 3, "give me 3d points"
        #            pcd = o3d.geometry.PointCloud()
        #            pcd.points = o3d.utility.Vector3dVector(pts)
        #            return pcd
        #        pcd = make_pcd(pc)
        #        things_to_print = [make_frame(), pcd]
        #
        #        o3d.visualization.draw_geometries(things_to_print)
        #
        #        import ipdb; ipdb.set_trace()

        pc_colors = np.stack(rgbs, axis=0)
        pc_colors = np.reshape(pc_colors, [-1, 3])
        pc_colors = pc_colors[selection, :]

        # extract the object pointcloud from table and object pointcloud

        # no, just get it from object box
        obj_xml = obj_info.obj_xml_file
        obj_xpos = env.env.sim.data.get_body_xpos('object0')
        obj_xmat = env.env.sim.data.get_body_xmat('object0')

        composed_xmat = np.eye(4, dtype=np.float32)
        composed_xmat[:3, 3] = obj_xpos
        composed_xmat[:3, :3] = np.reshape(obj_xmat, [3, 3])
        composed_xmat = np.dot(self.adam_T_origin, composed_xmat)
        object_xpos_adam =  composed_xmat[:3, 3] #np.dot(self.adam_T_origin, pcp_utils.geom.pts_addone(np.reshape(object_xpos, [1, 3])).T)[:3, 0]
        object_xmat_adam =  composed_xmat[:3, :3]#np.dot(self.adam_T_origin[:3, :3], np.reshape(object_xmat, [3, 3]))

        scale = obj_info.scale
        coords, combined_mesh = pcp_utils.np_vis.compute_bounding_box_from_obj_xml(
            obj_info.obj_xml_file, object_xpos_adam, object_xmat_adam, scale=scale, euler=obj_info.euler, return_combined_mesh=True)


        inliers_x = pc[:, 0] >= coords[0, 0]
        inliers_x2 = pc[:, 0] <= coords[-1, 0]
        inliers_y = pc[:, 1] >= coords[0, 1]
        inliers_y2 =  pc[:, 1] <= coords[-1, 1]
        inliers_z = pc[:, 2] >= coords[0, 2]
        inliers_z2 =  pc[:, 2] <= coords[-1, 2]
        inliers_table =  pc[:, 1] < -0.002

        selection = inliers_x & inliers_y & inliers_z & inliers_x2 & inliers_y2 & inliers_z2 & inliers_table

        object_pc = pc.copy()
        object_pc = object_pc[selection, :]

        if not canonicalize_pcs:
            # use the first camera as the frame 
            object_pc_one = pcp_utils.geom.pts_addone(object_pc)
            camR_T_camX = camR_T_camXs[0]
            camX_T_camR = np.linalg.inv(camR_T_camX)
            object_pc_X = np.matmul(camX_T_camR, object_pc_one.T).T[:,:3]



        #        import open3d as o3d
        #
        #
        #        def make_frame():
        #            return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #        def make_pcd(pts):
        #            assert pts.shape[1] == 3, "give me 3d points"
        #            pcd = o3d.geometry.PointCloud()
        #            pcd.points = o3d.utility.Vector3dVector(pts)
        #            return pcd
        #        pcd = make_pcd(object_pc)
        #        things_to_print = [make_frame(), pcd]
        #
        #        o3d.visualization.draw_geometries(things_to_print)



        #object_pc = self.extract_object_pcd(pc, obj_name=obj_name)



        if save:
            self.save_pcd(pc, save_dir, obj_name, pc_colors)

        if canonicalize_pcs:
            return object_pc, pc, pc_colors
        else:
            return object_pc, object_pc_X, camR_T_camX, pc, pc_colors 
    

    
    def sample_grasp_from_camera(self, env, obj, canonicalize_pcs=False):
        # output should be a dictionary
        # 'avg_reward': float32
        # 'success_rate': float32
        images = self.render_images(env)

        if self.config.data_collection_mode:
            # use only 3 frames
            ncams = images['rgb_camXs'].shape[0]
            vlist = list(range(ncams))

            if self.fix_view:
                vlist = list(range(ncams))
                idx = vlist
            else:
                np.random.shuffle(vlist)
                idx = vlist[:3]
            
            for item_name, item in images.items():
                images[item_name] = item[idx]

        images = self.convert_to_adam(images)

        if canonicalize_pcs:
            obj_pc, pc, pc_colors = self.preprocess_points_for_6dgrasp(env, obj, images, obj_name=obj.name, canonicalize_pcs=canonicalize_pcs)
        
        else:
            obj_pc_ori, obj_pc, camR_T_camX, pc, pc_colors = self.preprocess_points_for_6dgrasp(env, obj, images, obj_name=obj.name, canonicalize_pcs=canonicalize_pcs)

        if obj_pc.shape[0] == 0:
            obj_pc = np.zeros((1, 3), dtype=np.float32) 

        env_state = env.env.sim.get_state()

        latents = self.estimator.sample_latents()

        generated_grasps, generated_scores, _ = self.estimator.predict_grasps(
            self.sess,
            obj_pc,
            latents,
            num_refine_steps = self.num_refine_steps,
        )

        if not canonicalize_pcs: 
            # convert grasps and obj_pc back to canonical poses
            obj_pc_one = pcp_utils.geom.pts_addone(obj_pc)
            obj_pc = np.matmul(camR_T_camX, obj_pc_one.T).T[:,:3]
            generated_grasps = [np.matmul(camR_T_camX, grasp) for grasp in generated_grasps]

        output = {
            'generated_grasps': generated_grasps,
            'generated_scores': generated_scores,
            'pcd': pc,
            'pcd_color': pc_colors,
            'obj_pcd': obj_pc,
            'obj_pcd_ori': obj_pc_ori,
            'env_state': env_state,
        }

        output = self.filter_bad_grasps(output)


        if self.config.data_collection_mode:

            generated_grasps = output["generated_grasps"]
            # convert obj_pc and grasp point back
            obj_pc_one = pcp_utils.geom.pts_addone(obj_pc)
            obj_pc = np.matmul(np.linalg.inv(camR_T_camX), obj_pc_one.T).T[:,:3]
            output["obj_pcd_X"] = obj_pc

            generated_grasps = [np.matmul(np.linalg.inv(camR_T_camX), grasp) for grasp in generated_grasps]
            output["generated_grasps_X"] = generated_grasps
            output["camR_T_camX"] = camR_T_camX
        #save_file_path = os.path.join(self.output_grasps_dir, f'grasps_fn_{obj.name}.npy')
        #np.save(save_file_path, output)
        
        #import ipdb; ipdb.set_trace()

        return output
    def split_grasps(self, grasp):
        pos = grasp[:3, 3]
        orn = grasp[:3, :3]
        return pos, orn

    def filter_bad_grasps(self, data):
        num_grasps = len(data['generated_scores'])        
        keep_id = []
        for grasp_id in range(num_grasps):
            a_grasp = data['generated_grasps'][grasp_id]
            grasp_pos, grasp_orn = self.split_grasps(a_grasp)
            gripper_pc = np.matmul(self.gripper_pc_ori, grasp_orn.T)
            gripper_pc += grasp_pos[None]


            if np.max(gripper_pc, 0)[1] < 0: #going into table
                keep_id.append(grasp_id)

        data['generated_grasps'] = [data['generated_grasps'][id_] for id_ in keep_id]
        data['generated_scores'] = [data['generated_scores'][id_] for id_ in keep_id]

        print("#original_grasps:", num_grasps, "#grasps after filtering", len(data["generated_grasps"]))

        return data


