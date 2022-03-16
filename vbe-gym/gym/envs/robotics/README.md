# Robotics environments

Details and documentation on these robotics environments are available in our [blog post](https://blog.openai.com/ingredients-for-robotics-research/), the accompanying [technical report](https://arxiv.org/abs/1802.09464), and the [Gym website](https://gym.openai.com/envs/#robotics).

If you use these environments, please cite the following paper:

```
@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
```

## Fetch environments
<img src="https://openai.com/content/images/2018/02/fetch-reach.png" width="500">

[FetchReach-v0](https://gym.openai.com/envs/FetchReach-v0/): Fetch has to move its end-effector to the desired goal position.


<img src="https://openai.com/content/images/2018/02/fetch-slide.png" width="500">

[FetchSlide-v0](https://gym.openai.com/envs/FetchSlide-v0/): Fetch has to hit a puck across a long table such that it slides and comes to rest on the desired goal.


<img src="https://openai.com/content/images/2018/02/fetch-push.png" width="500">

[FetchPush-v0](https://gym.openai.com/envs/FetchPush-v0/): Fetch has to move a box by pushing it until it reaches a desired goal position.


<img src="https://openai.com/content/images/2018/02/fetch-pickandplace.png" width="500">

[FetchPickAndPlace-v0](https://gym.openai.com/envs/FetchPickAndPlace-v0/): Fetch has to pick up a box from a table using its gripper and move it to a desired goal above the table.

## Shadow Dexterous Hand environments
<img src="https://openai.com/content/images/2018/02/hand-reach.png" width="500">

[HandReach-v0](https://gym.openai.com/envs/HandReach-v0/): ShadowHand has to reach with its thumb and a selected finger until they meet at a desired goal position above the palm.


<img src="https://openai.com/content/images/2018/02/hand-block.png" width="500">

[HandManipulateBlock-v0](https://gym.openai.com/envs/HandManipulateBlock-v0/): ShadowHand has to manipulate a block until it achieves a desired goal position and rotation.


<img src="https://openai.com/content/images/2018/02/hand-egg.png" width="500">

[HandManipulateEgg-v0](https://gym.openai.com/envs/HandManipulateEgg-v0/): ShadowHand has to manipulate an egg until it achieves a desired goal position and rotation.


<img src="https://openai.com/content/images/2018/02/hand-pen.png" width="500">

[HandManipulatePen-v0](https://gym.openai.com/envs/HandManipulatePen-v0/): ShadowHand has to manipulate a pen until it achieves a desired goal position and rotation.

**Things specific to us:**

1. The environment can be loaded as follows ```env = gym.make('FetchPickAndPlace-v1')```. This will the load the environment with a default xml file which is mentioned in ```pick_and_place.py```

2. If one wants to load the environment with a specific ```xml_file``` which are all present in ```envs/robotics/assets/fetch``` execute 
the following command ```env = gym.make('FetchPickAndPlace-v1', xml_path=/path/to/xml/file, rim_pts=None)```

3. Rim points are points which lie on the rim of the object such as cups and bowls. If want to load with that ```env = gym.make('FetchPickAndPlace-v1', xml_path=/path/to/xml/file, rim_pts=/path/to/rim/points)```

Precomputed rim points are located in the following folder **envs/robotics/assets/fetch/rim_pts_dir**

If the environment is loaded with rim_pts, then an extra key in added to the observation dict called **gripper_to_rim**. This is a vector from current gripper position to current rim position.

All the mesh files for the objects are expected to lie in the following path: **envs/robotics/assets/stl/meshes**
