"""On import, install RoCo env/agent classes into PettingLLMs' lookup tables.

This is a side effect because Ray subprocesses import the package fresh and
need the registry populated before the engine reads it. Skipped silently in
environments that don't have PettingLLMs installed (e.g. the roco310 worker
env, where this package is also installed for its worker module).
"""


def register():
    try:
        from pettingllms.trainer import multiagentssys_register as R
    except ImportError:
        return
    from .env import RoCoEnv, RoCoEnvBatch
    from .worker import get_ray_roco_worker_cls
    from .agent import AliceAgent, BobAgent, ChadAgent

    R.ENV_CLASS_MAPPING["roco_env"] = RoCoEnv
    R.ENV_BATCH_CLASS_MAPPING["roco_env"] = RoCoEnvBatch
    R.ENV_WORKER_CLASS_MAPPING["roco_env"] = get_ray_roco_worker_cls
    R.AGENT_CLASS_MAPPING["Alice"] = AliceAgent
    R.AGENT_CLASS_MAPPING["Bob"] = BobAgent
    R.AGENT_CLASS_MAPPING["Chad"] = ChadAgent


register()
