import numpy as np

class Policy:
    def __init__(self, policy_name):
        self.policy_name = policy_name

    def run_forwards(self, env, num_rollouts, obj, path_length, render=True):
        # output should be a dictionary
        # 'avg_reward': float32
        # 'success_rate': float32
        raise NotImplementedError("Must be implemented in subclass.")

    def convert_robot_4dof_to_8dof(self, action):

        action_8  = np.zeros(8, )
        action_8[:3] = action[:3]
        action_8[3:7] = [1., 0., 1., 0.]
        action_8[-1] = action[-1]

        return action_8

    def env_render(self):
        if self.render:
             self.env.render()

        if self.save_video:

            img, depth = self.env.render_from_camera(512, 512, camera_name="front_view")
            #img = self.env.render(mode='rgb_array')
            self.imgs.append(np.rot90(img, k=3))


    def dump_video(self, filename):
        import imageio
        writer = imageio.get_writer(filename, fps=20)

        for im in self.imgs:
            writer.append_data(im)
        writer.close()

        #import ipdb; ipdb.set_trace()

        self.imgs = []