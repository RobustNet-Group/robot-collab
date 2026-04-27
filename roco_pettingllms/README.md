# roco_pettingllms

RoCo MuJoCo tasks as a PettingLLMs multi-agent RL environment. Trains a
shared-policy LLM (L1 prompt specialization, AT-GRPO) on the RoCoBench
collaboration tasks with sparse task-success reward.

## Layout

```
roco_pettingllms/
├── env.py          # RoCoEnv shells in the trainer process (no MuJoCo)
├── worker.py       # Ray actor that owns one MuJoCo physics per rollout
├── agent.py        # Alice/Bob/Chad dialog agents
└── config/
    ├── roco_L1_2agent.yaml   # pack, sweep, sandwich, rope
    └── roco_L1_3agent.yaml   # sort, cabinet
```

The worker is the only file that imports `rocobench`/`mujoco`/`dm_control`,
so the trainer process can stay on Python 3.12 with the PettingLLMs deps.

## Setup
```bash
# PettingLLMs venv
source pettingllms/pettingllms_venv/bin/activate
bash pettingllms/setup.bash
pip install -e roco_pettingllms
```

## Run       

```bash
# Launch training (PettingLLMs venv). This boots a local Ray head.
bash roco_pettingllms/scripts/train/roco/roco_L1_2agent.sh   # pack
# or
bash roco_pettingllms/scripts/train/roco/roco_L1_3agent.sh   # sort

# As soon as the trainer prints "Started a local Ray instance", join the
#    MuJoCo worker from a separate shell in the roco310 env. The custom
#    resource label is what schedules our actor onto this interpreter.
conda activate roco
cd ~/mavla
ray start --address=auto --resources='{"roco": 16}'
```

Override the task on the CLI:

```bash
bash roco_pettingllms/scripts/train/roco/roco_L1_2agent.sh env.task=sweep
```
