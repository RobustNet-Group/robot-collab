"""RoCo dialog agent. One class per role (Alice/Bob/Chad) since the
PettingLLMs engine looks up agent classes by name from turn_order.
"""
from pettingllms.multi_agent_env.base.agent import Agent

from .env import RoCoEnvState


class RoCoDialogAgent(Agent):
    agent_name = "Smith"

    def __init__(self, rollout_idx=None, **kwargs):
        super().__init__()
        self.rollout_idx = rollout_idx
        self._last_response = ""

    def reset(self):
        self._last_response = ""
        self.agent_reward = 0.0
        self.success = False

    def update_from_env(self, turn_idx, env_data):
        state: RoCoEnvState = env_data.state
        if not state.initialized:
            self.current_prompt = {"text": "Reply with: READY", "image": None}
            return
        body = state.agent_prompts.get(self.agent_name, "")
        chat = "\n".join(state.chat_history) if state.chat_history else ""
        feedback = state.last_feedback or ""
        text = body
        if chat:
            text += f"\n[Previous Chat]\n{chat}"
        if feedback:
            text += f"\n{feedback}"
        text += f"\nYou are {self.agent_name}, your response is:"
        self.current_prompt = {"text": text, "image": None}

    def update_from_model(self, response):
        self._last_response = response or ""

    def _is_final_speaker(self, env_data):
        order = env_data.config.multi_agent_interaction.turn_order
        return self.agent_name == order[-1]

    async def step(self, env_data, env_worker=None):
        state: RoCoEnvState = env_data.state

        if not state.initialized and env_worker is not None:
            new_state = await env_worker.reset.remote(state.task, state.seed)
            env_data.state = RoCoEnvState(**new_state)
            return

        msg = f"[{self.agent_name}]: {self._last_response}"
        state.chat_history = state.chat_history + [msg]

        if not self._is_final_speaker(env_data):
            return
        if "EXECUTE" not in self._last_response or env_worker is None:
            return

        full_dialog = "\n".join(state.chat_history)
        new_state = await env_worker.apply_plan.remote(
            full_dialog, list(state.chat_history)
        )
        env_data.state = RoCoEnvState(**new_state)
        env_data.done = env_data.state.done
        env_data.success = env_data.state.success

    def calculate_reward(self, env_data):
        self.agent_reward = float(env_data.state.reward)


class AliceAgent(RoCoDialogAgent):
    agent_name = "Alice"


class BobAgent(RoCoDialogAgent):
    agent_name = "Bob"


class ChadAgent(RoCoDialogAgent):
    agent_name = "Chad"
