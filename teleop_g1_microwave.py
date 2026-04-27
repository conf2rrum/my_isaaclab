from isaaclab.app import AppLauncher

# parser ...
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import g1_microwave_task  # noqa: F401
