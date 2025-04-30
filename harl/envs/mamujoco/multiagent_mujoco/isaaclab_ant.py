import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from scipy.spatial.transform import Rotation as R


class IsaacLabAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, "ant.xml", 5)
        utils.EzPickle.__init__(self)
        self.last_action = np.zeros(self.model.nu)  # store last action for obs
        for i in range(self.model.ngeom):
            self.model.geom_friction[i][:] = [1.0, 0.0, 0.0]

    def step(self, a):
        a *= 7.5
        self.last_action = a.copy()

        # Position before sim step (for progress reward)
        xposbefore = self.sim.data.qpos[0]

        # Simulation
        self.do_simulation(a, self.frame_skip)

        # Position after
        xposafter = self.sim.data.qpos[0]
        progress_reward = (xposafter - xposbefore)   # move along x

        # Alive reward
        alive_reward = 0.5 * self.dt

        # Orientation terms
        torso_id = self.model.body_name2id("torso")
        xmat = self.sim.data.body_xmat[torso_id].reshape(3, 3)
        z_axis = xmat[:, 2]  # base up vector
        x_axis = xmat[:, 0]  # base heading vector

        upright_proj = np.dot(z_axis, np.array([0, 0, 1]))
        upright_reward = (0.1 if upright_proj > 0.93 else 0.0) * self.dt

        # Move to target bonus (same target as Isaac Lab)
        target_pos = np.array([1000.0, 0.0])
        agent_xy = self.sim.data.qpos[:2]
        vec_to_target = target_pos - agent_xy
        dir_to_target = np.array([vec_to_target[0], vec_to_target[1], 0.0])
        dir_to_target /= np.linalg.norm(dir_to_target) + 1e-8

        heading_proj = np.dot(x_axis, dir_to_target)
        move_to_target_bonus = (0.5 if heading_proj > 0.8 else 0.0 ) * self.dt

        # Action L2 penalty
        action_l2_penalty = -0.005 * np.sum(np.square(a)) * self.dt

        # Energy penalty (|tau * qvel|), gear ratio ~15.0
        joint_vel = self.sim.data.qvel[6:]
        energy_penalty = -0.05 * np.sum(np.abs(a * joint_vel * 15.0)) * self.dt

        # Joint limit penalty (if near 0.99 of joint range)
        joint_pos = self.sim.data.qpos[7:]
        jnt_range = self.model.jnt_range
        normalized_pos = (joint_pos - jnt_range[1:, 0]) / (jnt_range[1:, 1] - jnt_range[1:, 0])
        near_limit = (normalized_pos < 0.01) | (normalized_pos > 0.99)
        joint_limit_penalty = -0.1 * np.sum(near_limit.astype(np.float32)) * self.dt

        # Total reward
        reward = (
            progress_reward
            + alive_reward
            + upright_reward
            + move_to_target_bonus
            + action_l2_penalty
            + energy_penalty
            + joint_limit_penalty
        )

        # Done check
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone

        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_progress=progress_reward,
                reward_alive=alive_reward,
                reward_upright=upright_reward,
                reward_heading=move_to_target_bonus,
                reward_action_l2=action_l2_penalty,
                reward_energy=energy_penalty,
                reward_joint_limits=joint_limit_penalty,
            ),
        )


    def _get_obs(self):
        torso_id = self.model.body_name2id("torso")
        xmat = self.sim.data.body_xmat[torso_id].reshape(3, 3)
        z_axis = xmat[:, 2]
        x_axis = xmat[:, 0]

        # base_height
        base_height = self.sim.data.qpos[2:3]

        # base_lin_vel
        base_lin_vel = self.sim.data.qvel[0:3]

        # base_ang_vel
        base_ang_vel = self.sim.data.qvel[3:6]

        # orientation: yaw & roll
        quat = self.sim.data.qpos[3:7]  # (w, x, y, z)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = r.as_euler("zyx")  # yaw, pitch, roll
        yaw, roll = euler[0], euler[2]
        base_yaw_roll = np.array([yaw, roll])

        # angle to target
        target_pos = np.array([1000.0, 0.0])
        agent_xy = self.sim.data.qpos[:2]
        vec_to_target = target_pos - agent_xy
        angle_to_target = np.arctan2(vec_to_target[1], vec_to_target[0]) - yaw
        angle_to_target = np.array([angle_to_target])

        # heading projection
        dir_to_target = np.array([vec_to_target[0], vec_to_target[1], 0.0])
        dir_to_target /= np.linalg.norm(dir_to_target) + 1e-8
        base_heading_proj = np.array([np.dot(x_axis, dir_to_target)])

        # base_up_proj
        base_up_proj = np.array([np.dot(z_axis, np.array([0, 0, 1]))])

        # joint positions (normalized)
        joint_pos = self.sim.data.qpos[7:]
        jnt_range = self.model.jnt_range
        joint_pos_norm = 2.0 * (joint_pos - jnt_range[1:, 0]) / (jnt_range[1:, 1] - jnt_range[1:, 0]) - 1.0

        # joint velocities (scaled relative)
        joint_vel = self.sim.data.qvel[6:] * 0.2

        # feet forces
        foot_names = ["front_left_leg", "front_right_leg", "back_leg", "right_back_leg"]
        foot_ids = [self.model.body_name2id(name) for name in foot_names]
        feet_forces = np.concatenate([
            np.clip(self.sim.data.cfrc_ext[bid], -1, 1) for bid in foot_ids
        ]) * 0.1

        # last action
        actions = self.last_action

        return np.concatenate([
            base_height,
            base_lin_vel,
            base_ang_vel,
            base_yaw_roll,
            angle_to_target,
            base_up_proj,
            base_heading_proj,
            joint_pos_norm,
            joint_vel,
            # feet_forces,
            actions,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        self.last_action = np.zeros(self.model.nu)  # clear stored action
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
