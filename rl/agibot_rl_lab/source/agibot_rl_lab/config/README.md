# AgiBot RL Lab

AgiBot X2 (lx2501 / T2.5, 29-DoF) RL locomotion，作为 IsaacLab extension 构建在
宇树 `unitree_rl_lab` 之上。

- **宇树部分** `unitree_rl_lab`：以 submodule（钉版本，只读）方式依赖，不修改。X2 仅
  复用其 `tasks.locomotion.mdp`（reward/event/observation/curriculum 函数）。
- **智元部分** `agibot_rl_lab`（本包）：X2 的 articulation（`assets/robots/agibot.py`）、
  资产（`x2_description/`）、速度任务（`tasks/locomotion/robots/x2/`）、PPO 配置
  （`tasks/locomotion/agents/rsl_rl_ppo_cfg.py`，rsl_rl 5.0 新 API）。

注册任务 id：`Agibot-X2-29dof-Velocity`。
