import embodied
import numpy as np

from . import gym


class LeggedRobot(embodied.Env):

    def __init__(
        self,
        task: str,
        repeat:int = 1,
        length: int = 1000,
        resets: bool = True,
        robot_type: str = None,
        enable_rendering: bool = False
    ):
        assert robot_type != None, "RobotType is None. Choose robot"
        assert robot_type in ("A1", "Go1"), "Incorrect robot type"
        assert task in ("sim", "real"), task

        # don't move the import from this place! It works only like this
        import motion_imitation.envs.env_builder as env_builder  

        self._gymenv = env_builder.build_env(
            enable_rendering=enable_rendering,
            num_action_repeat=repeat,
            use_real_robot=bool(task == "real"),
            robot_type = robot_type
        )
        self._env = gym.Gym(
            self._gymenv, obs_key="vector", act_key="action", checks=True
        )

    @property
    def obs_space(self):
        return {
            **self._env.obs_space,
            "image": embodied.Space(np.uint8, (64, 64, 3)),
        }

    @property
    def act_space(self):
        # return self._env.act_space
        return {
            "action": embodied.Space(np.float32, (12,), -1.0, 1.0),
            "reset": embodied.Space(bool, ()),
        }

    def step(self, action):
        obs = self._env.step(action)
        obs["image"] = self._gymenv.render("rgb_array")
        assert obs["image"].shape == (64, 64, 3), obs["image"].shape
        assert obs["image"].dtype == np.uint8, obs["image"].dtype
        return obs
