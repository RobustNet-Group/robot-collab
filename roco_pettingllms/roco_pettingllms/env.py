"""RoCo env shells in the trainer process. No MuJoCo here.

State and prompt text are produced worker-side and shipped over Ray.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pettingllms.multi_agent_env.base.env import Env


@dataclass
class RoCoEnvState:
    task: str = ""
    seed: int = 0
    step_idx: int = 0
    agent_prompts: Dict[str, str] = field(default_factory=dict)
    chat_history: List[str] = field(default_factory=list)
    last_feedback: Optional[str] = None
    last_response: Optional[str] = None
    last_executed_ok: bool = True
    reward: float = 0.0
    done: bool = False
    success: bool = False
    initialized: bool = False


class RoCoEnv(Env):
    def __init__(self, env_idx, rollout_idx, max_turns, config=None):
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, config=config)
        self.max_turns = max_turns
        self.state = RoCoEnvState()


class RoCoEnvBatch:
    def __init__(self, env_idx_list, env_indices, rollout_idx_list, samples,
                 max_turns, config, mode="train", *, env_workers=None):
        task = config.env.task
        env_indices = list(env_indices)
        if mode == "validate":
            env_indices = list(range(getattr(config.env, "validate_envs", 20)))

        self.env_list: List[RoCoEnv] = []
        for i, base_seed in enumerate(env_indices):
            seed_offset = 0 if mode == "train" else 100000
            seed = seed_offset + base_seed
            for s in range(samples):
                rollout_idx = i * samples + s
                env = RoCoEnv(env_idx=i, rollout_idx=rollout_idx,
                              max_turns=max_turns, config=config)
                env.state.task = task
                env.state.seed = seed
                self.env_list.append(env)
