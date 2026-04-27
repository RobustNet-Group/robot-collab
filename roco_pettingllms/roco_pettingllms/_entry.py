"""Entry point: register RoCo with PettingLLMs, then hand off to the trainer."""
import runpy

import roco_pettingllms

roco_pettingllms.register()
runpy.run_module("pettingllms.trainer.train", run_name="__main__")
