"""Ray actor that owns one MuJoCo physics instance per rollout.

Mirrors the shape of pettingllms.multi_agent_env.code.code_worker. Only this
file imports rocobench; the trainer process never touches MuJoCo.
"""
from typing import Any, Dict, List


def _build_stack(task: str, seed: int):
    from rocobench.envs import (
        SortOneBlockTask, CabinetTask, MoveRopeTask, SweepTask,
        MakeSandwichTask, PackGroceryTask,
    )
    from rocobench import MultiArmRRT
    from prompting.parser import LLMResponseParser
    from prompting.feedback import FeedbackManager

    task_map = {
        "sort": SortOneBlockTask, "cabinet": CabinetTask, "rope": MoveRopeTask,
        "sweep": SweepTask, "sandwich": MakeSandwichTask, "pack": PackGroceryTask,
    }
    output_mode = "action_and_path" if task in ("rope", "pack") else "action_only"
    control_freq = {"rope": 20, "pack": 10}.get(task, 15)

    env = task_map[task](
        render_freq=2000, image_hw=(400, 400), sim_forward_steps=300,
        error_freq=30, error_threshold=1e-5, randomize_init=True,
        render_point_cloud=0, render_cameras=["teaser"],
    )
    env.seed(np_seed=seed)
    env.reset(reload=True)
    robots = env.get_sim_robots()
    planner = MultiArmRRT(
        env.physics, robots=robots,
        graspable_object_names=env.get_graspable_objects(),
        allowed_collision_pairs=env.get_allowed_collision_pairs(),
    )
    keywords = ["NAME", "ACTION"] + (["PATH"] if output_mode == "action_and_path" else [])
    parser = LLMResponseParser(
        env, output_mode, env.robot_name_map, keywords,
        direct_waypoints=0, use_prepick=env.use_prepick,
        use_preplace=env.use_preplace, split_parsed_plans=False,
    )
    feedback = FeedbackManager(
        env=env, planner=planner, llm_output_mode=output_mode,
        robot_name_map=env.robot_name_map,
        step_std_threshold=env.waypoint_std_threshold, max_failed_waypoints=0,
    )
    policy_kwargs = dict(control_freq=control_freq, use_weld=True)
    return env, robots, planner, parser, feedback, policy_kwargs


def _agent_prompts(env) -> Dict[str, str]:
    obs = env.get_obs()
    action_desp = env.get_action_prompt()
    return {
        env.robot_name_map[r]: f"{action_desp}\n{env.get_agent_prompt(obs, env.robot_name_map[r])}"
        for r in env.robot_names
    }


def get_ray_roco_worker_cls(num_workers: int = 32):
    import ray

    cache_key = f"_cls_{num_workers}"
    if hasattr(get_ray_roco_worker_cls, cache_key):
        return getattr(get_ray_roco_worker_cls, cache_key)

    @ray.remote(num_cpus=1, max_concurrency=4, resources={"roco": 0.01})
    class _RoCoSimWorker:
        def __init__(self, idx: int):
            self.idx = int(idx)
            self.task = None
            self.seed = None
            self.env = None
            self.robots = None
            self.planner = None
            self.parser = None
            self.feedback = None
            self.policy_kwargs = None

        def reset(self, task: str, seed: int) -> Dict[str, Any]:
            (self.env, self.robots, self.planner, self.parser, self.feedback,
             self.policy_kwargs) = _build_stack(task, seed)
            self.task, self.seed = task, seed
            return self._state(chat_history=[])

        def apply_plan(self, response_text: str, chat_history: List[str]) -> Dict[str, Any]:
            from rocobench.policy import PlannedPathPolicy

            obs = self.env.get_obs()
            ok, reason, plans = self.parser.parse(obs, response_text)
            if not ok:
                return self._state(chat_history=chat_history,
                                   feedback=f"Parse failed: {reason}",
                                   response=response_text, executed_ok=False)

            for plan in plans:
                passed, fb = self.feedback.give_feedback(plan)
                if not passed:
                    return self._state(chat_history=chat_history, feedback=fb,
                                       response=response_text, executed_ok=False)

            for plan in plans:
                policy = PlannedPathPolicy(
                    physics=self.env.physics, robots=self.robots, path_plan=plan,
                    graspable_object_names=self.env.get_graspable_objects(),
                    allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                    plan_splitted=False, **self.policy_kwargs,
                )
                plan_ok, plan_reason = policy.plan(self.env)
                if not plan_ok:
                    return self._state(
                        chat_history=chat_history,
                        feedback=f"Motion plan failed: {plan_reason}",
                        response=response_text, executed_ok=False,
                    )
                while not policy.plan_exhausted:
                    action = policy.act(self.env.get_obs(), self.env.physics)
                    self.env.step(action, verbose=False)

            obs = self.env.get_obs()
            reward, done = self.env.get_reward_done(obs)
            return self._state(chat_history=[], response=response_text,
                               executed_ok=True, reward=float(reward),
                               done=bool(done), success=bool(reward > 0))

        def _state(self, *, chat_history, feedback=None, response=None,
                   executed_ok=True, reward=0.0, done=False, success=False):
            wrapped = (f"[Environment Feedback]\n{feedback}"
                       if feedback and not feedback.startswith("[Environment Feedback]")
                       else feedback)
            return dict(
                task=self.task, seed=self.seed, step_idx=0,
                agent_prompts=_agent_prompts(self.env),
                chat_history=list(chat_history),
                last_feedback=wrapped, last_response=response,
                last_executed_ok=executed_ok, reward=reward,
                done=done, success=success, initialized=True,
            )

    setattr(get_ray_roco_worker_cls, cache_key, _RoCoSimWorker)
    return _RoCoSimWorker
